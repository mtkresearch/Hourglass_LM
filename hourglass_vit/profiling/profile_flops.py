#!/usr/bin/env python3
"""
Compact ViT FLOPs table profiler.

This script profiles configurable timm ViT models and prints one compact row per
configuration, e.g.

================================================================================================================
Status   Size     Model                        MLP  Depth    Dim   Heads   Classes     GFLOPs     GMACs~  Params(M)
----------------------------------------------------------------------------------------------------------------
DONE     large    vit_large_patch16_224        0.5     22   1536      16       100     108.00      54.00     261.49
...

Supported:
  - vit sizes: tiny, small, base, large
  - custom mlp_ratio
  - custom depth / layers
  - custom embed_dim / dz
  - custom num_heads
  - custom num_classes, including None/headless
  - forward FLOPs by torch_flops
  - estimated training FLOPs = forward FLOPs x multiplier
  - measured forward+backward FLOPs by torch.utils.flop_counter.FlopCounterMode

Important:
  - TIMM_FUSED_ATTN=0 is set before importing timm.
  - torch_flops forward FLOPs count multiply and addition separately.
  - GMACs~ is displayed as GFLOPs / 2 for comparison with MAC-style papers.
"""

from __future__ import annotations

import os
os.environ["TIMM_FUSED_ATTN"] = "0"

import argparse
import itertools
import warnings
from typing import Any, Dict, Optional

import torch
import timm
from torch_flops import TorchFLOPsByFX

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


def profile_forward_torch_flops(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    device: str,
) -> int:
    model.eval()
    with torch.no_grad():
        _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        flops_counter = TorchFLOPsByFX(model)
        flops_counter.propagate(x)
        total_flops = flops_counter.print_total_flops(show=False)

    return int(total_flops)


def profile_train_measured_flops(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    num_classes: Optional[int],
    device: str,
) -> int:
    """
    Measure forward + backward FLOPs using PyTorch FlopCounterMode.

    If num_classes is not None, use CrossEntropyLoss with random labels.
    If num_classes is None/headless, use output.sum() as a scalar loss so the
    backbone backward graph is still exercised.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as e:
        raise RuntimeError(
            "torch.utils.flop_counter.FlopCounterMode is unavailable in this PyTorch build."
        ) from e

    model.train()
    model.zero_grad(set_to_none=True)

    with FlopCounterMode(display=False) as counter:
        out = model(x)

        if num_classes is not None:
            y = torch.randint(0, num_classes, (x.shape[0],), device=device)
            loss = torch.nn.functional.cross_entropy(out, y)
        else:
            loss = out.float().sum()

        loss.backward()

    return int(counter.get_total_flops())


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
    flops_mode: str,
    train_multiplier: float,
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

    params = count_params(model)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    if flops_mode == "forward":
        flops = profile_forward_torch_flops(model=model, x=x, device=device)
        forward_flops = flops
        measured_train_flops = None
    elif flops_mode == "train-estimate":
        forward_flops = profile_forward_torch_flops(model=model, x=x, device=device)
        flops = int(round(forward_flops * train_multiplier))
        measured_train_flops = None
    elif flops_mode == "train-measured":
        measured_train_flops = profile_train_measured_flops(
            model=model,
            x=x,
            num_classes=num_classes,
            device=device,
        )
        flops = measured_train_flops
        # Optional: also get forward FLOPs using the same model rebuilt fresh would be cleaner,
        # but we keep the compact mode fast and only report selected-mode GFLOPs.
        forward_flops = None
    else:
        raise ValueError(f"Unknown flops_mode: {flops_mode}")

    return {
        "vit_size": vit_size,
        "model_name": model_name,
        "mlp_ratio": mlp_ratio,
        "depth": depth,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_classes": num_classes,
        "flops": flops,
        "gflops": flops / 1e9,
        "gmacs_approx": flops / 2.0 / 1e9,
        "forward_flops": forward_flops,
        "measured_train_flops": measured_train_flops,
        "params_total": params["params_total"],
        "params_head": params["params_head"],
        "params_backbone": params["params_backbone"],
        "params_total_m": params["params_total"] / 1e6,
    }


def print_header(args: argparse.Namespace, num_classes: Optional[int]) -> None:
    print("=" * 112)
    print(f"FLOPs mode:       {args.flops_mode}")
    if args.flops_mode == "train-estimate":
        print(f"Train multiplier: {args.train_multiplier:g} x forward FLOPs")
    print(f"Device:           {args.device}")
    print(f"Input:            batch_size={args.batch_size}, image_size={args.image_size}x{args.image_size}")
    print(f"Classes:          {fmt_num_classes(num_classes)}")
    print(f"TIMM_FUSED_ATTN:  {os.environ.get('TIMM_FUSED_ATTN')}")
    print("=" * 112)
    print(
        f"{'Status':<8} {'Size':<8} {'Model':<25} {'MLP':>6} {'Depth':>6} "
        f"{'Dim':>6} {'Heads':>7} {'Classes':>9} {'GFLOPs':>10} {'GMACs~':>10} {'Params(M)':>10}"
    )
    print("-" * 112)


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
        f"{result['gflops']:>10.2f} "
        f"{result['gmacs_approx']:>10.2f} "
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
        f"{'-':>10} "
        f"{'-':>10} "
        f"{'-':>10}"
    )
    print(f"  {type(error).__name__}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact table profiler for configurable timm ViT FLOPs."
    )

    parser.add_argument(
        "--vit-sizes",
        nargs="+",
        default=["tiny"],
        choices=list(DEFAULT_VIT_CONFIGS.keys()),
        help="ViT sizes to profile. Example: --vit-sizes tiny base large.",
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

    parser.add_argument(
        "--flops-mode",
        choices=["forward", "train-estimate", "train-measured"],
        default="forward",
        help=(
            "forward: torch_flops forward FLOPs. "
            "train-estimate: torch_flops forward FLOPs multiplied by --train-multiplier. "
            "train-measured: PyTorch FlopCounterMode forward+backward FLOPs."
        ),
    )
    parser.add_argument(
        "--train-multiplier",
        type=float,
        default=3.0,
        help="Multiplier for --flops-mode train-estimate. Default: 3.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_classes = parse_optional_int(args.num_classes)
    num_classes_label = fmt_num_classes(num_classes)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or check CUDA_VISIBLE_DEVICES.")

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
                    flops_mode=args.flops_mode,
                    train_multiplier=args.train_multiplier,
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
                    error=e,
                )

            finally:
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    print("-" * 112)
    print("Done.")
    print("=" * 112)


if __name__ == "__main__":
    main()
