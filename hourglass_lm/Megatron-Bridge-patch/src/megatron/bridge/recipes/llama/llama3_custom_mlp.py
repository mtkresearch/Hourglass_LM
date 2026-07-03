"""Recipes for Llama 3.2 with custom multi-layer MLP architecture.

This module provides pre-configured training recipes for Llama 3.2 models
using custom MLP architectures with multiple internal layers and residual connections.

Example:
    >>> from megatron.bridge.recipes.llama.llama3_custom_mlp import llama32_1b_custom_mlp_pretrain_config_local
    >>> cfg = llama32_1b_custom_mlp_pretrain_config_local()
    >>> cfg.model.num_mlp_layers = 5  # Use 5-layer MLP
    >>> cfg.model.mlp_intermediate_sizes = [8192, 6144, 4096, 6144, 8192]
"""

import torch

from megatron.bridge.models.llama.llama_provider import Llama32ModelProvider1BWithCustomMLP
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer

# OLMo2 tokenizer vocabulary size
OLMO2_VOCAB_SIZE: int = 100278


def llama32_custom_mlp_pretrain_config_local() -> ConfigContainer:
    """Return a pre-training config for Llama 3.2 with hourglass MLP.

    Training from scratch with OLMo2-tokenized data (no HF pretrained model needed).

    Architecture: Llama 3.2 with hourglass MLP featuring:
    - Multiple sequential MLP layers 
    - Residual connections between MLP layers
    - Customizable intermediate dimensions for each layer


    Returns:
        ConfigContainer: Complete training configuration with custom MLP provider
    """
    cfg = _pretrain_common()

    # Use the custom MLP model provider directly (training from scratch)
    cfg.model = Llama32ModelProvider1BWithCustomMLP()

    # Set vocab_size to match OLMo2 tokenizer
    cfg.model.vocab_size = OLMO2_VOCAB_SIZE
    cfg.model.should_pad_vocab = True

    # Tokenizer - NullTokenizer since data is already pre-tokenized by OLMo2
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = OLMO2_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Set to ["path/to/data"] for real data
    cfg.dataset.num_workers = 8
    cfg.dataset.sequence_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096

    # Custom MLP settings - adjust these for your architecture
    cfg.model.num_mlp_layers = 1  # Number of internal MLP layers
    cfg.model.mlp_use_residual_proj = True 
    cfg.model.mlp_intermediate_sizes = [cfg.model.ffn_hidden_size] * cfg.model.num_mlp_layers

    # Training config
    cfg.train.train_iters = 1168251
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1
    cfg.train.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 2000

    # Logger config
    cfg.logger.log_timers_to_tensorboard = True

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 500

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg