#!/usr/bin/env python3
"""
Compact ViT latency benchmark table.

Measures real wall-clock model latency in ms/image for configurable timm ViT
models. This complements FLOPs profiling: FLOPs are hardware-independent
operation counts, while latency depends on GPU, dtype, batch size, kernels,
memory bandwidth, torch/timm versions, and compilation.

Default mode:
  inference forward only, using CUDA Events if CUDA is used.

Example:
  python3 benchmark_vit_latency_table.py \
    --vit-sizes large \
    --mlp-ratios 0.5 \
    --embed-dims 1536 \
    --depths 22 23 24 25 26 \
    --num-heads 16 \
    --num-classes 100 \
    --batch-size 32 \
    --device cuda:0 \
    --dtype fp16

Output:
============================================================================================================================
Status   Size     Model                        MLP  Depth    Dim   Heads   Classes  Batch  DType       ms/img      img/s  Total ms  Params(M)
----------------------------------------------------------------------------------------------------------------------------
DONE     large    vit_large_patch16_224        0.5     25   1536      16       100     32   fp16        ...
----------------------------------------------------------------------------------------------------------------------------

Notes:
  - For inference latency, larger batch sizes usually give lower ms/image but
    higher end-to-end batch latency. Report both batch size and ms/image.
  - CUDA timing is asynchronous; this script uses CUDA Events and synchronize().
  - For CPU timing, this script uses perf_counter with warmup and repeats.
  - --mode train-step measures forward + loss + backward + optimizer step.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
import warnings
from typing import Any, Dict, Optional

# Keep attention visible/unfused only if you want to match FLOPs tracing behavior.
# For latency, fused attention may be faster. Default here lets the user choose.
if "TIMM_FUSED_ATTN" not in os.environ:
    os.environ["TIMM_FUSED_ATTN"] = "1"

import torch
import timm

warnings.filterwarnings("ignore")


DEFAULT_VIT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "model_name": "vit_tiny_patch16_224",
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "small": {
        "model_name": "vit_small_patch16_224",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "base": {
        "model_name": "vit_base_patch16_224",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large": {
        "model_name": "vit_large_patch16_224",
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
    },
}


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value

    text = str(value).strip().lower()
    if text in {"none", "null", "no", "headless", "no_head", "no-head"}:
        return None

    parsed = int(text)
    if parsed < 0:
        raise ValueError(f"num_classes must be non-negative or None, got {value}.")
    return parsed


def fmt_num_classes(num_classes: Optional[int]) -> str:
    return "None" if num_classes is None else str(num_classes)


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = 0
    for name, p in model.named_parameters():
        if p.requires_grad and name.startswith("head."):
            head += p.numel()
    return {
        "params_total": total,
        "params_head": head,
        "params_backbone": total - head,
    }


def dtype_from_arg(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_vit_model(
    *,
    model_name: str,
    image_size: int,
    num_classes: Optional[int],
    mlp_ratio: float,
    depth: int,
    embed_dim: int,
    num_heads: int,
) -> torch.nn.Module:
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads, got embed_dim={embed_dim}, "
            f"num_heads={num_heads}."
        )

    effective_num_classes = 0 if num_classes is None else num_classes

    return timm.create_model(
        model_name,
        pretrained=False,
        img_size=image_size,
        num_classes=effective_num_classes,
        mlp_ratio=mlp_ratio,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
    )


def maybe_autocast(device: str, dtype: str):
    if not device.startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    if dtype == "fp32":
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype_from_arg(dtype), enabled=True)


def run_inference_step(model: torch.nn.Module, x: torch.Tensor, device: str, dtype: str):
    with torch.no_grad(), maybe_autocast(device, dtype):
        return model(x)


def run_train_step(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_classes: Optional[int],
    device: str,
    dtype: str,
):
    optimizer.zero_grad(set_to_none=True)
    with maybe_autocast(device, dtype):
        out = model(x)
        if num_classes is None:
            loss = out.float().sum()
        else:
            loss = torch.nn.functional.cross_entropy(out.float(), y)
    loss.backward()
    optimizer.step()
    return loss


def benchmark_cuda_events(
    *,
    step_fn,
    warmup: int,
    iters: int,
    device: str,
) -> float:
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        step_fn()
    end.record()

    torch.cuda.synchronize()
    total_ms = float(start.elapsed_time(end))
    return total_ms / iters


def benchmark_cpu_perf_counter(
    *,
    step_fn,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        step_fn()

    start = time.perf_counter()
    for _ in range(iters):
        step_fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def profile_one(
    *,
    vit_size: str,
    model_name: str,
    image_size: int,
    batch_size: int,
    num_classes: Optional[int],
    mlp_ratio: float,
    depth: int,
    embed_dim: int,
    num_heads: int,
    device: str,
    dtype: str,
    mode: str,
    warmup: int,
    iters: int,
    channels_last: bool,
    compile_model: bool,
) -> Dict[str, Any]:
    model = build_vit_model(
        model_name=model_name,
        image_size=image_size,
        num_classes=num_classes,
        mlp_ratio=mlp_ratio,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
    ).to(device)
    model.eval() if mode == "inference" else model.train()

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if compile_model:
        model = torch.compile(model)

    params = count_params(model)

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    y = None
    optimizer = None
    if mode == "train-step":
        if num_classes is not None:
            y = torch.randint(0, num_classes, (batch_size,), device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if mode == "inference":
        step_fn = lambda: run_inference_step(model, x, device, dtype)
    elif mode == "train-step":
        step_fn = lambda: run_train_step(
            model=model,
            x=x,
            y=y,
            optimizer=optimizer,
            num_classes=num_classes,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if device.startswith("cuda"):
        avg_batch_ms = benchmark_cuda_events(
            step_fn=step_fn,
            warmup=warmup,
            iters=iters,
            device=device,
        )
        peak_mem_bytes = torch.cuda.max_memory_allocated()
    else:
        avg_batch_ms = benchmark_cpu_perf_counter(
            step_fn=step_fn,
            warmup=warmup,
            iters=iters,
        )
        peak_mem_bytes = None

    ms_per_image = avg_batch_ms / batch_size
    images_per_second = batch_size * 1000.0 / avg_batch_ms

    return {
        "vit_size": vit_size,
        "model_name": model_name,
        "mlp_ratio": mlp_ratio,
        "depth": depth,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "dtype": dtype,
        "mode": mode,
        "avg_batch_ms": avg_batch_ms,
        "ms_per_image": ms_per_image,
        "images_per_second": images_per_second,
        "params_total": params["params_total"],
        "params_head": params["params_head"],
        "params_backbone": params["params_backbone"],
        "params_total_m": params["params_total"] / 1e6,
        "peak_mem_mb": None if peak_mem_bytes is None else peak_mem_bytes / (1024 ** 2),
    }


def print_header(args: argparse.Namespace, num_classes: Optional[int]) -> None:
    print("=" * 124)
    print(f"Latency mode:     {args.mode}")
    print(f"Device:           {args.device}")
    print(f"Input:            batch_size={args.batch_size}, image_size={args.image_size}x{args.image_size}")
    print(f"DType:            {args.dtype}")
    print(f"Classes:          {fmt_num_classes(num_classes)}")
    print(f"Warmup / iters:   {args.warmup} / {args.iters}")
    print(f"channels_last:    {args.channels_last}")
    print(f"torch.compile:    {args.compile}")
    print(f"TIMM_FUSED_ATTN:  {os.environ.get('TIMM_FUSED_ATTN')}")
    print("=" * 124)
    print(
        f"{'Status':<8} {'Size':<8} {'Model':<25} {'MLP':>6} {'Depth':>6} "
        f"{'Dim':>6} {'Heads':>7} {'Classes':>9} {'Batch':>6} {'DType':>6} "
        f"{'ms/img':>10} {'img/s':>10} {'Total ms':>10} {'Params(M)':>10}"
    )
    print("-" * 124)


def print_row(status: str, result: Dict[str, Any], num_classes_label: str) -> None:
    print(
        f"{status:<8} "
        f"{result['vit_size']:<8} "
        f"{result['model_name']:<25} "
        f"{result['mlp_ratio']:>6g} "
        f"{result['depth']:>6} "
        f"{result['embed_dim']:>6} "
        f"{result['num_heads']:>7} "
        f"{num_classes_label:>9} "
        f"{result['batch_size']:>6} "
        f"{result['dtype']:>6} "
        f"{result['ms_per_image']:>10.4f} "
        f"{result['images_per_second']:>10.2f} "
        f"{result['avg_batch_ms']:>10.3f} "
        f"{result['params_total_m']:>10.2f}"
    )


def print_error_row(
    *,
    vit_size: str,
    model_name: str,
    mlp_ratio: float,
    depth: int,
    embed_dim: int,
    num_heads: int,
    num_classes_label: str,
    batch_size: int,
    dtype: str,
    error: Exception,
) -> None:
    print(
        f"{'ERROR':<8} "
        f"{vit_size:<8} "
        f"{model_name:<25} "
        f"{mlp_ratio:>6g} "
        f"{depth:>6} "
        f"{embed_dim:>6} "
        f"{num_heads:>7} "
        f"{num_classes_label:>9} "
        f"{batch_size:>6} "
        f"{dtype:>6} "
        f"{'-':>10} "
        f"{'-':>10} "
        f"{'-':>10} "
        f"{'-':>10}"
    )
    print(f"  {type(error).__name__}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact table latency benchmark for configurable timm ViT models."
    )

    parser.add_argument(
        "--vit-sizes",
        nargs="+",
        default=["tiny"],
        choices=list(DEFAULT_VIT_CONFIGS.keys()),
        help="ViT sizes to benchmark. Example: --vit-sizes tiny base large.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override timm model name for all runs. Usually leave unset.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--num-classes",
        "--class-num",
        dest="num_classes",
        default="100",
        help="Number of classes. Use None/null/headless to exclude classification head.",
    )

    parser.add_argument("--mlp-ratios", nargs="+", type=float, default=None)
    parser.add_argument("--embed-dims", nargs="+", type=int, default=None)
    parser.add_argument("--depths", nargs="+", type=int, default=None)
    parser.add_argument("--num-heads", nargs="+", type=int, default=None)

    parser.add_argument("--mode", choices=["inference", "train-step"], default="inference")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile(model).")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on CUDA for fp32 benchmark.")
    parser.add_argument(
        "--disable-timm-fused-attn",
        action="store_true",
        help="Set TIMM_FUSED_ATTN=0 before model creation. Usually slower but aligns with FLOPs tracing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_classes = parse_optional_int(args.num_classes)
    num_classes_label = fmt_num_classes(num_classes)

    if args.disable_timm_fused_attn:
        os.environ["TIMM_FUSED_ATTN"] = "0"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or check CUDA_VISIBLE_DEVICES.")

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    print_header(args, num_classes)

    for vit_size in args.vit_sizes:
        defaults = DEFAULT_VIT_CONFIGS[vit_size]

        model_name = args.model_name if args.model_name is not None else defaults["model_name"]
        mlp_ratios = args.mlp_ratios if args.mlp_ratios is not None else [defaults["mlp_ratio"]]
        embed_dims = args.embed_dims if args.embed_dims is not None else [defaults["embed_dim"]]
        depths = args.depths if args.depths is not None else [defaults["depth"]]
        num_heads_list = args.num_heads if args.num_heads is not None else [defaults["num_heads"]]

        for mlp_ratio, depth, embed_dim, num_heads in itertools.product(
            mlp_ratios, depths, embed_dims, num_heads_list
        ):
            try:
                result = profile_one(
                    vit_size=vit_size,
                    model_name=model_name,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    num_classes=num_classes,
                    mlp_ratio=mlp_ratio,
                    depth=depth,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    device=args.device,
                    dtype=args.dtype,
                    mode=args.mode,
                    warmup=args.warmup,
                    iters=args.iters,
                    channels_last=args.channels_last,
                    compile_model=args.compile,
                )
                print_row("DONE", result, num_classes_label)

            except Exception as e:
                print_error_row(
                    vit_size=vit_size,
                    model_name=model_name,
                    mlp_ratio=mlp_ratio,
                    depth=depth,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_classes_label=num_classes_label,
                    batch_size=args.batch_size,
                    dtype=args.dtype,
                    error=e,
                )

            finally:
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    print("-" * 124)
    print("Done.")
    print("=" * 124)


if __name__ == "__main__":
    main()
