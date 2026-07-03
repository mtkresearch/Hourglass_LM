# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom MLP implementations for Llama models with multi-layer MLP support."""

import copy
import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    import transformer_engine.pytorch as te
    HAS_TRANSFORMER_ENGINE = True
except ImportError:
    HAS_TRANSFORMER_ENGINE = False


class MultiLayerMLP(MegatronModule):
    """Multi-layer MLP with dedicated layernorm and residual connections.
    
    Architecture:
    - input from attention  
    - For each MLP layer i:
        - residual_in = input (or previous layer's output for i > 0)
        - output = layernorm_i(residual_in)
        - mlp_out = mlp_i(output)
        - Prepare residual for next layer: residual_in + mlp_out
    
    Each sub-layer supports SwiGLU, tensor parallelism, and configurable intermediate sizes.
    
    Args:
        config: Transformer configuration (includes tensor_model_parallel_size, etc.)
        moe: Whether to apply MoE (for compatibility)
        moe_expert_model_parallel_size: MoE config (for compatibility)
    """
    
    def __init__(self, config, moe=False, moe_expert_model_parallel_size=1):
        super().__init__(config)
        
        self.config = config
        self.moe = moe
        
        # Number of stacked MLP sub-layers
        self.num_mlp_layers = getattr(config, 'num_mlp_layers', 3)
        
        # Default intermediate size from config
        ffn_hidden_size = getattr(config, 'ffn_hidden_size', config.hidden_size * 4)
        
        # Per-layer intermediate sizes for "wide-narrow-wide" patterns
        # Example: [8192, 1024, 8192] instead of uniform [8192, 8192, 8192]
        mlp_inter_sizes = getattr(config, 'mlp_intermediate_sizes', None)
        if mlp_inter_sizes is None:
            mlp_inter_sizes = [ffn_hidden_size] * self.num_mlp_layers
        self.mlp_intermediate_sizes = mlp_inter_sizes
        
        # Get LOCAL spec MLP submodules (without built-in layernorm)
        backend = LocalSpecProvider()
        mlp_spec = get_mlp_module_spec_for_backend(backend=backend)
        
        # Build MLP sub-layers and their preceding layernorms
        self.layernorms = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        
        for i in range(self.num_mlp_layers):
            # Create RMSNorm for each layer based on config normalization
            if getattr(config, 'normalization', 'RMSNorm') == 'RMSNorm':
                layernorm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'layernorm_epsilon', 1e-6))
            else:
                layernorm = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layernorm_epsilon', 1e-6))
            self.layernorms.append(layernorm)
            
            # Add MLP sub-layer
            mlp_layer = self._build_single_mlp(i, mlp_spec.submodules)
            self.mlp_layers.append(mlp_layer)
        
        # Dropout for bias-dropout-add
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.0))
    
    def _build_single_mlp(self, layer_idx: int, submodules: MLPSubmodules) -> MLP:
        """Build a single MLP sub-layer using Megatron-Core's local MLP.
        
        Creates a per-layer config copy with the appropriate ffn_hidden_size.
        Uses LOCAL spec (no TE modules, no built-in layernorm).
        
        Args:
            layer_idx: Index of this sub-layer (determines intermediate size)
            submodules: MLP submodule specs (linear_fc1, linear_fc2, activation)
        
        Returns:
            Megatron-Core MLP instance with per-layer ffn_hidden_size
        """
        inter_size = self.mlp_intermediate_sizes[layer_idx]
        
        # Create a config copy with this layer's ffn_hidden_size
        layer_config = copy.copy(self.config)
        layer_config.ffn_hidden_size = inter_size
        
        return MLP(
            config=layer_config,
            submodules=submodules,
            ffn_hidden_size=inter_size,
        )
    
    def forward(self, hidden_states, **kwargs):
        """Forward pass through multi-layer MLP with layernorm and residuals.
        
        Architecture for each layer i:
        1. residual_in = output from previous layer (or input for i=0)
        2. output = layernorm_i(residual_in)
        3. mlp_out, mlp_bias = mlp_i(output)
        4. mlp_out = mlp_out + mlp_bias (if bias present)
        5. mlp_out = dropout(mlp_out)
        6. output = residual_in + mlp_out  (residual connection)
        
        This preparation for next layer's residual, unless it's the last layer.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] from attention
            **kwargs: Additional arguments (padding_mask, etc.) - unused
        
        Returns:
            tuple: (output, None) for compatibility with TransformerLayer's mlp_bda=IdentityOp
        """
        residual = hidden_states
        
        for i, (layernorm, mlp_layer) in enumerate(zip(self.layernorms, self.mlp_layers)):
            # Apply layernorm to residual
            normalized = layernorm(residual)
            
            # Apply MLP
            mlp_out, mlp_bias = mlp_layer(normalized)
            
            # Add bias if present (None when add_bias_linear=False)
            if mlp_bias is not None:
                mlp_out = mlp_out + mlp_bias
            
            # Apply dropout
            mlp_out = self.dropout(mlp_out)
            
            # Residual connection: prepare input for next layer (or final output)
            residual = residual + mlp_out
        
        # Return (output
        return residual