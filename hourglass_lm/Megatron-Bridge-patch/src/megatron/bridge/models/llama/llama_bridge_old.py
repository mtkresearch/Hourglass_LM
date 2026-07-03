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

import logging
from typing import Optional

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import LlamaForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel, model_type="llama")
class LlamaBridge(MegatronModelBridge):
    """
    Megatron Bridge for Llama Causal LM.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace Llama config to Megatron GPTModelProvider.

        Uses base class implementation for common conversion, then sets
        Llama-specific config and enables RoPE scaling for Llama 3.1/3.2 models.

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the Llama config

        Returns:
            GPTModelProvider configured for Llama architecture
        """
        provider = super().provider_bridge(hf_pretrained)

        # Llama-specific Megatron defaults
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.hidden_dropout = 0.0
        provider.bias_activation_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.bias_dropout_fusion = True
        provider.apply_rope_fusion = True
        provider.rotary_percent = 1.0

        # Enable RoPE scaling for Llama 3.1/3.2 models via Megatron Core's built-in support
        hf_config = hf_pretrained.config
        hf_rope_scaling = getattr(hf_config, "rope_scaling", None)
        if hf_rope_scaling is not None and hf_rope_scaling.get("rope_type") == "llama3":
            provider.rope_scaling = True
            provider.rope_scaling_factor = hf_rope_scaling.get("factor", 8.0)

        return provider

    def provider_bridge_with_custom_mlp(
        self,
        hf_pretrained: PreTrainedCausalLM,
        num_mlp_layers: int = 3,
        mlp_intermediate_sizes: Optional[list] = None,
        mlp_use_residual_proj: bool = True,
    ) -> GPTModelProvider:
        """Convert HuggingFace Llama config to custom MLP provider.

        Creates a provider configured for custom multi-layer MLP instead of
        standard single-layer FFN.

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the Llama config
            num_mlp_layers: Number of MLP layers (default: 3)
            mlp_intermediate_sizes: Custom intermediate sizes per layer (optional)
            mlp_use_residual_proj: Whether to use residual connections (default: True)

        Returns:
            Llama32ModelProvider1BWithCustomMLP configured for training
        """
        from megatron.bridge.models.llama.llama_provider import Llama32ModelProvider1BWithCustomMLP

        # Get base provider with standard configuration
        base_provider = self.provider_bridge(hf_pretrained)

        # Create custom MLP provider from base configuration
        # This elegantly copies all fields and only replaces custom MLP parameters
        custom_mlp_provider = Llama32ModelProvider1BWithCustomMLP.from_base_provider(
            base_provider=base_provider,
            num_mlp_layers=num_mlp_layers,
            mlp_intermediate_sizes=mlp_intermediate_sizes,
            mlp_use_residual_proj=mlp_use_residual_proj,
        )

        logger.info(f"Created custom MLP provider with {num_mlp_layers} layers")

        return custom_mlp_provider

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        """Convert Megatron GPTModelProvider config to HuggingFace Llama config dict.

        Uses base class implementation, then adds RoPE scaling for Llama 3.1/3.2.
        Supports both standard single-layer MLP and custom multi-layer MLP.

        Args:
            provider: GPTModelProvider with Llama configuration

        Returns:
            Dictionary of HuggingFace LlamaConfig parameters
        """
        hf_config = super(LlamaBridge, cls).megatron_to_hf_config(provider)

        # Handle RoPE scaling for Llama 3.1/3.2 models
        if provider.rope_scaling:
            hf_config["rope_scaling"] = {
                "rope_type": "llama3",
                "factor": provider.rope_scaling_factor,
                # Use Megatron Core defaults for these values
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            }

        return hf_config

    @classmethod
    def extract_mlp_weights_for_custom(
        cls,
        megatron_state_dict: dict,
        num_mlp_layers: int,
        hidden_size: int = 2048,
    ) -> dict:
        """Extract and consolidate multi-layer MLP weights back to HF single-layer format.

        For custom MLP models, this function:
        1. Extracts weights from all MLP layers
        2. Averages or selects the first layer's weights
        3. Converts back to HF's up_proj/down_proj format

        Args:
            megatron_state_dict: Megatron model state dict
            num_mlp_layers: Number of MLP layers in custom MLP
            hidden_size: Hidden dimension of the model

        Returns:
            Updated state dict suitable for HF model
        """
        hf_state_dict = megatron_state_dict.copy()

        # Iterate through transformer layers
        num_layers = max(
            int(k.split('.')[2]) for k in megatron_state_dict.keys()
            if k.startswith('decoder.layers.')
        ) + 1

        for layer_idx in range(num_layers):
            # Megatron custom MLP structure
            mlp_prefix = f"decoder.layers.{layer_idx}.mlp"

            # Check if this is a custom MLP layer
            first_mlp_fc1_name = f"{mlp_prefix}.mlp_layers.0.0.weight"
            if first_mlp_fc1_name not in megatron_state_dict:
                # Not custom MLP, keep original structure
                continue

            # Extract first layer weights (representative of the MLP)
            fc1_weight = megatron_state_dict[first_mlp_fc1_name]
            fc2_weight = megatron_state_dict[f"{mlp_prefix}.mlp_layers.0.2.weight"]

            # Average all layers' weights (optional: could use just first layer)
            for layer_in_mlp in range(1, num_mlp_layers):
                layer_fc1_name = f"{mlp_prefix}.mlp_layers.{layer_in_mlp}.0.weight"
                layer_fc2_name = f"{mlp_prefix}.mlp_layers.{layer_in_mlp}.2.weight"

                if layer_fc1_name in megatron_state_dict:
                    fc1_weight = fc1_weight + megatron_state_dict[layer_fc1_name]
                if layer_fc2_name in megatron_state_dict:
                    fc2_weight = fc2_weight + megatron_state_dict[layer_fc2_name]

            fc1_weight = fc1_weight / num_mlp_layers
            fc2_weight = fc2_weight / num_mlp_layers

            # Convert back to HF format (should be transposed if needed)
            hf_prefix = f"model.layers.{layer_idx}.mlp"

            # In Megatron, linear weights are [out, in], in HF they're [out, in]
            # So direct assignment should work, but may need transpose
            if f"{hf_prefix}.up_proj.weight" in hf_state_dict or f"{hf_prefix}.gate_proj.weight" in hf_state_dict:
                # Split fc1_weight back into gate and up projections
                mid_size = fc1_weight.shape[0] // 2
                if f"{hf_prefix}.gate_proj.weight" in hf_state_dict:
                    hf_state_dict[f"{hf_prefix}.gate_proj.weight"] = fc1_weight[:mid_size]
                    hf_state_dict[f"{hf_prefix}.up_proj.weight"] = fc1_weight[mid_size:]
                else:
                    hf_state_dict[f"{hf_prefix}.up_proj.weight"] = fc1_weight

                if f"{hf_prefix}.down_proj.weight" in hf_state_dict:
                    hf_state_dict[f"{hf_prefix}.down_proj.weight"] = fc2_weight

            # Remove custom MLP weights from state dict
            for layer_in_mlp in range(num_mlp_layers):
                layer_fc1_name = f"{mlp_prefix}.mlp_layers.{layer_in_mlp}.0.weight"
                layer_fc2_name = f"{mlp_prefix}.mlp_layers.{layer_in_mlp}.2.weight"
                layer_fc1_bias = f"{mlp_prefix}.mlp_layers.{layer_in_mlp}.0.bias"
                layer_fc2_bias = f"{mlp_prefix}.mlp_layers.{layer_in_mlp}.2.bias"

                for param_name in [layer_fc1_name, layer_fc2_name, layer_fc1_bias, layer_fc2_bias]:
                    if param_name in hf_state_dict:
                        del hf_state_dict[param_name]

        return hf_state_dict

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",  # te implementation
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",  # local implementation
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",  # te implementation
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",  # local implementation
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

    def initialize_custom_mlp_from_hf(
        self,
        megatron_model: GPTModel,
        hf_model: LlamaForCausalLM,
        num_mlp_layers: int = 3,
    ) -> None:
        """Initialize custom multi-layer MLP from HF single-layer FFN weights.

        For custom MLP models with multiple layers, this function:
        1. Maps HF's up_proj/down_proj to the first MLP layer
        2. Replicates or randomly initializes subsequent layers
        3. Properly handles residual projections

        Args:
            megatron_model: Megatron model with MultiLayerMLP
            hf_model: HuggingFace Llama model
            num_mlp_layers: Number of custom MLP layers in the model
        """
        logger.info(f"Initializing custom MLP with {num_mlp_layers} layers from HF weights")

        hf_dict = dict(hf_model.named_parameters())
        megatron_dict = dict(megatron_model.named_parameters())

        # Get number of transformer layers
        hf_num_layers = len(hf_model.model.layers)
        megatron_num_layers = len(megatron_model.module.decoder.layers if hasattr(megatron_model, 'module') else megatron_model.decoder.layers)

        for layer_idx in range(hf_num_layers):
            # HF layer structure
            hf_up_proj_name = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            hf_down_proj_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            hf_gate_proj_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"

            if hf_up_proj_name not in hf_dict:
                logger.warning(f"Cannot find {hf_up_proj_name} in HF model, skipping layer {layer_idx}")
                continue

            # Megatron custom MLP layer structure
            megatron_layer_prefix = f"decoder.layers.{layer_idx}.mlp"

            # Get HF weights
            hf_up_proj = hf_dict[hf_up_proj_name]  # [ffn_hidden, hidden]
            hf_down_proj = hf_dict[hf_down_proj_name]  # [hidden, ffn_hidden]
            hf_gate_proj = hf_dict.get(hf_gate_proj_name)  # [ffn_hidden, hidden] or None

            # Concatenate gate and up for gated MLP
            if hf_gate_proj is not None:
                hf_fc1_weight = torch.cat([hf_gate_proj, hf_up_proj], dim=0)
            else:
                hf_fc1_weight = hf_up_proj

            # Initialize first MLP layer with HF weights
            first_mlp_fc1_name = f"{megatron_layer_prefix}.mlp_layers.0.0.weight"  # First layer's first linear
            first_mlp_fc2_name = f"{megatron_layer_prefix}.mlp_layers.0.2.weight"  # First layer's second linear

            if first_mlp_fc1_name in megatron_dict:
                # Transpose if needed for different weight conventions
                megatron_dict[first_mlp_fc1_name].copy_(hf_fc1_weight.t() if hf_fc1_weight.shape[0] != megatron_dict[first_mlp_fc1_name].shape[0] else hf_fc1_weight)
                logger.debug(f"Initialized {first_mlp_fc1_name}")

            if first_mlp_fc2_name in megatron_dict:
                megatron_dict[first_mlp_fc2_name].copy_(hf_down_proj.t() if hf_down_proj.shape[0] != megatron_dict[first_mlp_fc2_name].shape[0] else hf_down_proj)
                logger.debug(f"Initialized {first_mlp_fc2_name}")

            # For subsequent layers, copy the first layer weights (or could random initialize)
            for layer_in_mlp in range(1, num_mlp_layers):
                layer_fc1_name = f"{megatron_layer_prefix}.mlp_layers.{layer_in_mlp}.0.weight"
                layer_fc2_name = f"{megatron_layer_prefix}.mlp_layers.{layer_in_mlp}.2.weight"

                if layer_fc1_name in megatron_dict and first_mlp_fc1_name in megatron_dict:
                    # Copy from first layer (alternatively could use random init)
                    megatron_dict[layer_fc1_name].copy_(megatron_dict[first_mlp_fc1_name])
                    logger.debug(f"Initialized {layer_fc1_name} from first layer")

                if layer_fc2_name in megatron_dict and first_mlp_fc2_name in megatron_dict:
                    megatron_dict[layer_fc2_name].copy_(megatron_dict[first_mlp_fc2_name])
                    logger.debug(f"Initialized {layer_fc2_name} from first layer")

        logger.info("Custom MLP initialization complete")
