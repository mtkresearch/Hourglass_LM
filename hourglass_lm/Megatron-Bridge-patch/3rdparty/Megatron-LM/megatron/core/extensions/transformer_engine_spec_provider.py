# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from typing import Optional, Tuple

import torch.nn as nn
import transformer_engine as te  # type: ignore[import-untyped]

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
    _get_extra_te_kwargs,
)
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.utils import get_te_version, is_te_min_version


class _TERMSNormOnly(nn.Module):
    """Wrapper to force RMSNorm usage, inheriting nn.Module for proper PyTorch integration."""
    
    def __init__(self, config, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.rms_norm = te.pytorch.RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            sequence_parallel=config.sequence_parallel,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
            **_get_extra_te_kwargs(config),
        )
    
    def forward(self, x):
        """Forward pass - delegate to RMSNorm."""
        return self.rms_norm(x)
    
    def __getattr__(self, name):
        """Delegate attribute access to RMSNorm for compatibility with TENorm interface."""
        # Avoid infinite recursion by checking nn.Module attributes first
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.rms_norm, name)


class TESpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return TEColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module TE backend uses"""
        return TERowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return TELayerNormColumnParallelLinear

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module to use for layer norm
        
        Args:
            rms_norm: Force RMSNorm usage when True
            for_qk: For Q/K layer norm in attention
        
        Returns:
            A norm class that will be instantiated later by build_module()
        """
        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used
            # for QKLayerNorm if TE Version < 1.9;
            # we instead use the Apex implementation.
            return FusedLayerNorm
        
        # Return RMSNorm-forcing wrapper if requested, otherwise TENorm with config-based selection
        return _TERMSNormOnly if rms_norm else TENorm

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return TEDotProductAttention

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        if (
            moe_use_grouped_gemm
            and TEColumnParallelGroupedLinear is not None
            and not moe_use_legacy_grouped_gemm
        ):
            return TEGroupedMLP, MLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
            )
        elif moe_use_grouped_gemm:
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
            return GroupedMLP, None
        else:
            if not is_te_min_version("1.7.0.dev0"):
                warnings.warn(
                    "Only transformer-engine>=1.7.0 supports MoE experts, "
                    f"but your version is {get_te_version()}. "
                    "Use local linear implementation instead."
                )
                return SequentialMLP, MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                )
            return SequentialMLP, MLPSubmodules(
                linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            )

    def activation_func(self) -> type:
        """Which module to use for activation function"""
        return TEActivationOp
