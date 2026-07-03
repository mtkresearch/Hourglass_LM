#!/usr/bin/env python3
"""
Example: Training Llama 3.2 1B with custom multi-layer MLP architecture.

This script shows how to use the custom MLP provider to train Llama models
with advanced MLP modifications:
1. Multiple sequential MLP layers (instead of single FFN)
2. Residual connections between layers
3. Customizable intermediate dimensions

Usage:
    Basic (training from scratch with OLMo2 data):
        python custom_mlp_pretraining_example.py

    With YAML config file:
        python custom_mlp_pretraining_example.py \
            --config-file conf/custom_mlp_config.yaml

    With CLI overrides (takes precedence over YAML):
        python custom_mlp_pretraining_example.py \
            model.num_mlp_layers=4 \
            model.mlp_intermediate_sizes=[8192,1024,1024,8192]

    Combining YAML and CLI:
        python custom_mlp_pretraining_example.py \
            --config-file conf/custom_mlp_config.yaml \
            train.train_iters=10000
"""

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml

from megatron.bridge.recipes.llama.llama3_custom_mlp import (
    llama32_custom_mlp_pretrain_config_local
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
from megatron.core.datasets.utils import get_blend_from_list


logger = logging.getLogger(__name__)


def _normalize_data_path_list(value: Any) -> list[str] | None:
    """Normalize YAML dataset path field to list[str] expected by get_blend_from_list."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise TypeError(f"Unsupported dataset path type: {type(value)}")


def _apply_per_split_dataset_from_yaml(config: ConfigContainer, config_filepath: str | None) -> None:
    """Map YAML train/valid/test_data_path into dataset.blend_per_split for GPT pretraining."""
    if not config_filepath:
        return

    with open(config_filepath, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}

    dataset_cfg = raw_cfg.get("dataset") or {}
    train_data_path = _normalize_data_path_list(dataset_cfg.get("train_data_path"))
    valid_data_path = _normalize_data_path_list(dataset_cfg.get("valid_data_path"))
    test_data_path = _normalize_data_path_list(dataset_cfg.get("test_data_path"))

    if not any([train_data_path, valid_data_path, test_data_path]):
        return

    config.dataset.blend = None
    config.dataset.blend_per_split = [
        get_blend_from_list(train_data_path),
        get_blend_from_list(valid_data_path),
        get_blend_from_list(test_data_path),
    ]
    config.dataset.split = None

    print("[DEBUG] Applied per-split dataset from YAML:")
    print(f"[DEBUG]   train_data_path={train_data_path}")
    print(f"[DEBUG]   valid_data_path={valid_data_path}")
    print(f"[DEBUG]   test_data_path={test_data_path}")
    print(f"[DEBUG]   blend_per_split={config.dataset.blend_per_split}")


def parse_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments and CLI overrides."""
    parser = argparse.ArgumentParser(
        description="Train Llama 3.2 1B with custom multi-layer MLP",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML config file (optional). CLI overrides take precedence."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Separate known args from CLI overrides
    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def main() -> None:
    """Run pretraining with YAML configuration and CLI overrides."""
    args, cli_overrides = parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Debug: Print CLI overrides
    print(f"\n[DEBUG] CLI overrides: {cli_overrides}")
    
    # Load base configuration from recipe (training from scratch, no HF model needed)
    print(f"Loading Llama 3.2 1B custom MLP recipe (from scratch)...")
    config: ConfigContainer = llama32_custom_mlp_pretrain_config_local()
    
    # Debug: Check config before processing overrides
    print(f"[DEBUG] Before overrides - mlp_intermediate_sizes: {config.model.mlp_intermediate_sizes}")
    
    # Process YAML config file and CLI overrides
    # Priority (highest to lowest):
    # 1. Command-line overrides
    # 2. YAML config file
    # 3. Base recipe defaults
    config = process_config_with_overrides(
        config,
        config_filepath=args.config_file,
        cli_overrides=cli_overrides or None,
    )

    # YAML keys like train_data_path/valid_data_path are not fields on GPTDatasetConfig,
    # so map them into blend_per_split explicitly for independent train/valid datasets.
    if not config.dataset.blend:
        _apply_per_split_dataset_from_yaml(config, args.config_file)
    
    # Debug: Check config after processing overrides
    print(f"[DEBUG] After overrides - mlp_intermediate_sizes: {config.model.mlp_intermediate_sizes}")
    print(f"[DEBUG] Config type: {type(config.model.mlp_intermediate_sizes)}")
    
    # Convert dataset.blend from simple list to MCore tuple format if needed.
    # YAML/CLI sets blend as ["path1", "path2"] (plain list),
    # but MCore expects (["path1", "path2"], None) — a Tuple[List[str], Optional[List[float]]].
    # Also supports weight+path format: ["0.7", "path1", "0.3", "path2"]
    if config.dataset.blend is not None and isinstance(config.dataset.blend, list):
        config.dataset.blend = get_blend_from_list(config.dataset.blend)
        print(f"[DEBUG] Converted blend to MCore format: {config.dataset.blend}")
    
    # Log configuration
    print(f"\n=== MLP Configuration ===")
    print(f"Number of MLP layers: {config.model.num_mlp_layers}")
    print(f"Use residual connections: {config.model.mlp_use_residual_proj}")
    if hasattr(config.model, 'mlp_intermediate_sizes') and config.model.mlp_intermediate_sizes is not None:
        print(f"Intermediate sizes: {config.model.mlp_intermediate_sizes}")
    else:
        print(f"Intermediate sizes: [ffn_hidden_size={config.model.ffn_hidden_size}] * {config.model.num_mlp_layers}")
    print(f"\n=== Training Configuration ===")
    print(f"Training iterations: {config.train.train_iters}")
    print(f"Global batch size: {config.train.global_batch_size}")
    print(f"Sequence length: {config.dataset.seq_length}")
    print()
    
    # Build OLMo eval callback if requested.
    callbacks: List = []

    # Start training
    pretrain(config, forward_step_func=forward_step, callbacks=callbacks or None)


if __name__ == "__main__":
    main()
