#!/usr/bin/env python3
"""
Search ViT configurations that match a default/conventional ViT in parameter count
and training FLOPs while changing the MLP ratio.

Compared with a formula-only parameter explorer, this script builds the actual
timm ViT model and computes FLOPs with profiler logic.

Training FLOPs modes:
  1) train-estimate:
       forward FLOPs are computed by torch_flops, then multiplied by --train-multiplier.
       Default multiplier = 3.0.

  2) train-measured:
       forward + loss + backward FLOPs are measured with
       torch.utils.flop_counter.FlopCounterMode.
       This is NOT forward FLOPs x 3.

This lets you search configs such as:
  - fixed depth = 12
  - fixed depth = 11
  - target mlp_ratio = 0.5
  - search embed_dim such that training FLOPs are close to default conventional ViT

Example:
  python3 search_vit_config_by_training_flops.py \
    --target-size tiny \
    --mlp-ratio 0.5 \
    --depths 12 11 \
    --num-classes 196 \
    --flops-mode train-measured \
    --device cuda:0

Note:
  - TIMM_FUSED_ATTN=0 is set before importing timm for torch_flops tracing.
  - train-measured uses PyTorch FlopCounterMode and may have a slightly different
    FLOP-counting convention from torch_flops. Use one mode consistently across
    baseline and candidates.
  - The reported d_h is read from the actual timm model. This aligns with timm's
    int(embed_dim * mlp_ratio) behavior and does not use round().
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TIMM_FUSED_ATTN"] = "0"

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


def infer_actual_mlp_hidden_dim(model: torch.nn.Module, embed_dim: int, mlp_ratio: float) -> int:
    """
    Infer the actual MLP hidden dimension used by timm.

    timm's ViT MLP hidden size is created from int(embed_dim * mlp_ratio)
    inside the model constructor. Reading fc1.out_features from the actual
    model is safer than recomputing it, because it always matches the model
    that was really built.

    Example:
      embed_dim=294, mlp_ratio=0.25
      294 * 0.25 = 73.5
      timm actual dh = int(73.5) = 73
    """
    try:
        return int(model.blocks[0].mlp.fc1.out_features)
    except Exception:
        # Fallback aligned with timm behavior; do not use round().
        return int(embed_dim * mlp_ratio)


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

        counter = TorchFLOPsByFX(model)
        counter.propagate(x)
        forward_flops = counter.print_total_flops(show=False)

    return int(forward_flops)


def profile_train_measured_flops(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    num_classes: Optional[int],
    device: str,
) -> int:
    """
    Measure forward + backward FLOPs using PyTorch FlopCounterMode.

    If num_classes is not None:
      use CrossEntropyLoss with random labels.
    If num_classes is None/headless:
      use output.sum() as a scalar loss so backward is still exercised.
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


def profile_config(
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
    also_forward: bool,
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

    forward_flops: Optional[int] = None
    measured_train_flops: Optional[int] = None

    if flops_mode == "train-estimate":
        forward_flops = profile_forward_torch_flops(model=model, x=x, device=device)
        training_flops = int(round(forward_flops * train_multiplier))

    elif flops_mode == "train-measured":
        measured_train_flops = profile_train_measured_flops(
            model=model,
            x=x,
            num_classes=num_classes,
            device=device,
        )
        training_flops = measured_train_flops

        if also_forward:
            # Rebuild model to avoid any possible state/grad side effects from backward.
            model2 = build_vit_model(
                model_name=model_name,
                image_size=image_size,
                num_classes=num_classes,
                mlp_ratio=mlp_ratio,
                depth=depth,
                embed_dim=embed_dim,
                num_heads=num_heads,
            ).to(device)
            x2 = torch.randn(batch_size, 3, image_size, image_size, device=device)
            forward_flops = profile_forward_torch_flops(model=model2, x=x2, device=device)
            del model2, x2

    else:
        raise ValueError(f"Unknown flops_mode: {flops_mode}")

    # Align displayed/search-recorded dh with the actual timm model.
    # Do not use round(embed_dim * mlp_ratio): timm truncates via int(...).
    # Better: read the value from the model itself.
    dh = infer_actual_mlp_hidden_dim(model, embed_dim, mlp_ratio)
    dh_float = embed_dim * mlp_ratio

    return {
        "vit_size": vit_size,
        "model_name": model_name,
        "mlp_ratio": mlp_ratio,
        "depth": depth,
        "embed_dim": embed_dim,
        "dz": embed_dim,
        "mlp_hidden_dim": dh,
        "dh": dh,
        "dh_from_ratio_float": dh_float,
        "dh_timm_int": int(dh_float),
        "dh_python_round": int(round(dh_float)),
        "num_heads": num_heads,
        "head_dim": embed_dim // num_heads,
        "num_classes": num_classes,
        "image_size": image_size,
        "batch_size": batch_size,
        "flops_mode": flops_mode,
        "forward_flops": forward_flops,
        "forward_gflops": None if forward_flops is None else forward_flops / 1e9,
        "training_flops": training_flops,
        "training_gflops": training_flops / 1e9,
        "training_gmacs_approx": training_flops / 2.0 / 1e9,
        "measured_train_flops": measured_train_flops,
        "train_multiplier": train_multiplier,
        "params_total": params["params_total"],
        "params_head": params["params_head"],
        "params_backbone": params["params_backbone"],
        "params_total_m": params["params_total"] / 1e6,
        "params_head_m": params["params_head"] / 1e6,
        "params_backbone_m": params["params_backbone"] / 1e6,
    }


def default_depths_for_size(vit_size: str) -> List[int]:
    d = int(DEFAULT_VIT_CONFIGS[vit_size]["depth"])
    return [d, max(1, d - 1)]


def make_embed_dim_grid(
    *,
    base_dim: int,
    min_dim: Optional[int],
    max_dim: Optional[int],
    step: int,
    divisor: int,
    include_default_scales: bool,
) -> List[int]:
    if min_dim is None:
        min_dim = max(divisor, int(round(base_dim * 0.5)))
    if max_dim is None:
        max_dim = int(round(base_dim * 2.25))

    if min_dim > max_dim:
        raise ValueError(f"embed-min > embed-max: {min_dim} > {max_dim}")

    dims = set()

    # Exhaustive search over multiples of divisor; optionally thin by embed-step.
    for d in range(min_dim, max_dim + 1):
        if d % divisor != 0:
            continue
        if step > 1 and d % step != 0:
            continue
        dims.add(d)

    if include_default_scales:
        for s in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]:
            raw = int(round(base_dim * s))
            adjusted = int(round(raw / divisor) * divisor)
            if min_dim <= adjusted <= max_dim:
                dims.add(adjusted)

    return sorted(dims)


def add_diffs(
    candidate: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    train_upper_pct: float = 0.0,
    params_upper_pct: float = 0.0,
) -> Dict[str, Any]:
    train_diff_abs = candidate["training_flops"] - baseline["training_flops"]
    train_diff_pct = train_diff_abs / baseline["training_flops"] * 100.0

    params_diff_abs = candidate["params_total"] - baseline["params_total"]
    params_diff_pct = params_diff_abs / baseline["params_total"] * 100.0

    train_upper = baseline["training_flops"] * (1.0 + train_upper_pct / 100.0)
    params_upper = baseline["params_total"] * (1.0 + params_upper_pct / 100.0)

    candidate["train_diff_pct"] = train_diff_pct
    candidate["params_diff_pct"] = params_diff_pct
    candidate["train_diff_abs"] = train_diff_abs
    candidate["params_diff_abs"] = params_diff_abs

    # Exact-under flags are useful for status display.
    candidate["train_leq_baseline"] = candidate["training_flops"] <= baseline["training_flops"]
    candidate["params_leq_baseline"] = candidate["params_total"] <= baseline["params_total"]

    # within_* flags include the optional upper tolerance.
    candidate["train_upper_pct"] = train_upper_pct
    candidate["params_upper_pct"] = params_upper_pct
    candidate["within_train_target"] = candidate["training_flops"] <= train_upper
    candidate["within_params_target"] = candidate["params_total"] <= params_upper
    return candidate


def rank_candidates(
    candidates: List[Dict[str, Any]],
    *,
    target_depth: int,
    require_train_within: bool,
    require_params_within: bool,
    prefer_under_target: bool,
) -> List[Dict[str, Any]]:
    filtered = []
    for c in candidates:
        if require_train_within and not c["within_train_target"]:
            continue
        if require_params_within and not c["within_params_target"]:
            continue
        filtered.append(c)

    def key(c: Dict[str, Any]) -> Tuple[float, float, float, int]:
        depth_gap = abs(c["depth"] - target_depth)

        if prefer_under_target:
            # Closest below/equal training target is best.
            # Above-target points are only shown when allowed, but penalized
            # unless include-more-flops switches prefer_under_target off.
            if c["train_diff_pct"] <= 0:
                train_gap = -c["train_diff_pct"]
            else:
                train_gap = 1e9 + c["train_diff_pct"]
        else:
            # Closest to baseline from either direction is best.
            train_gap = abs(c["train_diff_pct"])

        param_gap = abs(c["params_diff_pct"])
        dim = c["embed_dim"]
        return (depth_gap, train_gap, param_gap, dim)

    return sorted(filtered, key=key)


def print_baseline(baseline: Dict[str, Any]) -> None:
    fwd = "--" if baseline["forward_gflops"] is None else f"{baseline['forward_gflops']:.2f}"

    print("\nBaseline/default conventional config")
    print("=" * 138)
    print(
        f"{'Size':<8} {'Model':<25} {'MLP':>6} {'Depth':>6} {'Dim':>6} "
        f"{'dh':>6} {'Heads':>7} {'Classes':>9} {'Train GFLOPs':>14} "
        f"{'Fwd GFLOPs':>12} {'Params(M)':>10}"
    )
    print("-" * 138)
    print(
        f"{baseline['vit_size']:<8} {baseline['model_name']:<25} "
        f"{baseline['mlp_ratio']:>6g} {baseline['depth']:>6} "
        f"{baseline['embed_dim']:>6} {baseline['dh']:>6} "
        f"{baseline['num_heads']:>7} {fmt_num_classes(baseline['num_classes']):>9} "
        f"{baseline['training_gflops']:>14.2f} {fwd:>12} "
        f"{baseline['params_total_m']:>10.2f}"
    )
    print("=" * 138)


def print_candidate_table(candidates: List[Dict[str, Any]], *, title: str, top_k: int) -> None:
    print(f"\n{title}")
    print("=" * 154)
    print(
        f"{'Rank':>4} {'Status':<8} {'Size':<8} {'Model':<25} {'MLP':>6} "
        f"{'Depth':>6} {'Dim':>6} {'dh':>6} {'Heads':>7} {'Classes':>9} "
        f"{'Train GFLOPs':>14} {'ΔTrain%':>9} {'Params(M)':>10} {'ΔParam%':>9}"
    )
    print("-" * 154)

    if not candidates:
        print("No candidate found under the current constraints.")
        print("=" * 154)
        return

    for i, c in enumerate(candidates[:top_k], start=1):
        if c["train_leq_baseline"] and c["params_leq_baseline"]:
            status = "OK"
        elif c["within_train_target"] and c["within_params_target"]:
            status = "OVER"
        else:
            status = "CHECK"
        print(
            f"{i:>4} {status:<8} {c['vit_size']:<8} {c['model_name']:<25} "
            f"{c['mlp_ratio']:>6g} {c['depth']:>6} {c['embed_dim']:>6} "
            f"{c['dh']:>6} {c['num_heads']:>7} {fmt_num_classes(c['num_classes']):>9} "
            f"{c['training_gflops']:>14.2f} {c['train_diff_pct']:>+9.2f} "
            f"{c['params_total_m']:>10.2f} {c['params_diff_pct']:>+9.2f}"
        )

    print("=" * 154)


def atomic_save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search ViT embed_dim/depth configs matching baseline training FLOPs and params."
    )

    parser.add_argument(
        "--target-size",
        "--vit-size",
        dest="target_size",
        choices=list(DEFAULT_VIT_CONFIGS.keys()),
        default="tiny",
        help="Baseline/default ViT size to match. Default: tiny.",
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        required=True,
        help="Candidate MLP ratio to search, e.g. 0.5.",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=None,
        help="Candidate depths. Default: baseline depth and baseline depth - 1.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Candidate number of heads. Default: same as baseline.",
    )
    parser.add_argument(
        "--num-classes",
        "--class-num",
        dest="num_classes",
        default="196",
        help="Number of classes. Use None/null/headless to exclude classification head. Default: 196.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument(
        "--flops-mode",
        choices=["train-estimate", "train-measured"],
        default="train-estimate",
        help=(
            "train-estimate: torch_flops forward FLOPs x --train-multiplier. "
            "train-measured: FlopCounterMode measured forward+backward FLOPs."
        ),
    )
    parser.add_argument(
        "--train-multiplier",
        type=float,
        default=3.0,
        help="Used only by --flops-mode train-estimate. Default: 3.0.",
    )
    parser.add_argument(
        "--also-forward",
        action="store_true",
        help=(
            "When --flops-mode train-measured, also compute torch_flops forward GFLOPs. "
            "This is slower but useful for debugging."
        ),
    )

    parser.add_argument(
        "--embed-min",
        type=int,
        default=None,
        help="Minimum candidate embed_dim. Default: 0.5 x baseline dim.",
    )
    parser.add_argument(
        "--embed-max",
        type=int,
        default=None,
        help="Maximum candidate embed_dim. Default: 2.25 x baseline dim.",
    )
    parser.add_argument(
        "--embed-step",
        type=int,
        default=1,
        help=(
            "Optional coarse step for embed_dim search. Candidates must also satisfy divisor. "
            "Use 1 for exhaustive multiples of divisor. Default: 1."
        ),
    )
    parser.add_argument(
        "--divisor-mode",
        choices=["heads", "2heads", "custom"],
        default="2heads",
        help=(
            "Embedding dimension constraint. 'heads': divisible by heads; "
            "'2heads': divisible by 2*heads so head_dim is even; "
            "'custom': use --embed-divisor."
        ),
    )
    parser.add_argument("--embed-divisor", type=int, default=None)
    parser.add_argument(
        "--include-default-scales",
        action="store_true",
        help="Also include scaled dims such as 0.5x, 0.75x, 1.5x baseline dim.",
    )

    parser.add_argument(
        "--allow-train-exceed",
        action="store_true",
        help=(
            "Legacy option: allow training FLOPs to exceed baseline without an upper bound; "
            "rank by absolute closeness. Usually prefer --include-more-flops."
        ),
    )
    parser.add_argument(
        "--include-more-flops",
        action="store_true",
        help=(
            "Include candidates that are slightly above the baseline training FLOPs. "
            "The upper bound is controlled by --more-flops-pct. Disabled by default."
        ),
    )
    parser.add_argument(
        "--more-flops-pct",
        type=float,
        default=2.0,
        help=(
            "When --include-more-flops is enabled, allow training FLOPs up to "
            "baseline x (1 + pct/100). Default: 2.0."
        ),
    )
    parser.add_argument(
        "--more-params-pct",
        type=float,
        default=2.0,
        help=(
            "When --require-params-leq is enabled, allow params up to "
            "baseline x (1 + pct/100). Default: 2.0."
        ),
    )
    parser.add_argument(
        "--require-params-leq",
        action="store_true",
        help=(
            "Also require candidate params <= baseline. If --include-more-flops is set, "
            "--more-params-pct provides a small upper tolerance. Default: False."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_classes = parse_optional_int(args.num_classes)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or check CUDA_VISIBLE_DEVICES.")

    defaults = DEFAULT_VIT_CONFIGS[args.target_size]
    model_name = defaults["model_name"]
    baseline_embed_dim = int(defaults["embed_dim"])
    baseline_depth = int(defaults["depth"])
    baseline_heads = int(defaults["num_heads"])
    baseline_mlp_ratio = float(defaults["mlp_ratio"])

    candidate_heads = args.num_heads if args.num_heads is not None else baseline_heads

    if args.divisor_mode == "heads":
        embed_divisor = candidate_heads
    elif args.divisor_mode == "2heads":
        embed_divisor = candidate_heads * 2
    else:
        if args.embed_divisor is None:
            raise ValueError("--embed-divisor is required when --divisor-mode custom.")
        embed_divisor = args.embed_divisor

    depths = args.depths if args.depths is not None else default_depths_for_size(args.target_size)

    print("=" * 138)
    print("ViT config search by training FLOPs and params")
    print(f"Target size:       {args.target_size}")
    print(f"Baseline model:    {model_name}")
    print(f"Baseline config:   dim={baseline_embed_dim}, depth={baseline_depth}, heads={baseline_heads}, mlp={baseline_mlp_ratio}")
    print(f"Candidate mlp:     {args.mlp_ratio}")
    print(f"Candidate depths:  {depths}")
    print(f"Candidate heads:   {candidate_heads}")
    print(f"Embed divisor:     {embed_divisor}")
    print(f"Num classes:       {fmt_num_classes(num_classes)}")
    print(f"FLOPs mode:        {args.flops_mode}")
    if args.flops_mode == "train-estimate":
        print(f"Training FLOPs:    torch_flops forward FLOPs x {args.train_multiplier:g}")
    else:
        print("Training FLOPs:    FlopCounterMode measured forward + backward FLOPs")
    if args.include_more_flops:
        print(f"Extra FLOPs:       enabled, upper tolerance = +{args.more_flops_pct:g}%")
        if args.require_params_leq:
            print(f"Extra params:      enabled, upper tolerance = +{args.more_params_pct:g}%")
    elif args.allow_train_exceed:
        print("Extra FLOPs:       unbounded exceedance allowed by --allow-train-exceed")
    else:
        print("Extra FLOPs:       disabled, candidates must not exceed baseline training FLOPs")
    print(f"Device:            {args.device}")
    print(f"TIMM_FUSED_ATTN:   {os.environ.get('TIMM_FUSED_ATTN')}")
    print("=" * 138)

    baseline = profile_config(
        vit_size=args.target_size,
        model_name=model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_classes=num_classes,
        mlp_ratio=baseline_mlp_ratio,
        depth=baseline_depth,
        embed_dim=baseline_embed_dim,
        num_heads=baseline_heads,
        device=args.device,
        flops_mode=args.flops_mode,
        train_multiplier=args.train_multiplier,
        also_forward=args.also_forward,
    )
    print_baseline(baseline)

    dims = make_embed_dim_grid(
        base_dim=baseline_embed_dim,
        min_dim=args.embed_min,
        max_dim=args.embed_max,
        step=args.embed_step,
        divisor=embed_divisor,
        include_default_scales=args.include_default_scales,
    )

    print(f"\nSearching {len(dims)} embed_dim values x {len(depths)} depths = {len(dims) * len(depths)} candidates...")

    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []

    for depth, embed_dim in itertools.product(depths, dims):
        try:
            result = profile_config(
                vit_size=args.target_size,
                model_name=model_name,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_classes=num_classes,
                mlp_ratio=args.mlp_ratio,
                depth=depth,
                embed_dim=embed_dim,
                num_heads=candidate_heads,
                device=args.device,
                flops_mode=args.flops_mode,
                train_multiplier=args.train_multiplier,
                also_forward=args.also_forward,
            )
            train_upper_pct = args.more_flops_pct if args.include_more_flops else 0.0
            params_upper_pct = (
                args.more_params_pct if (args.include_more_flops and args.require_params_leq) else 0.0
            )
            add_diffs(
                result,
                baseline,
                train_upper_pct=train_upper_pct,
                params_upper_pct=params_upper_pct,
            )
            candidates.append(result)

        except Exception as e:
            errors.append(f"depth={depth}, embed_dim={embed_dim}: {type(e).__name__}: {e}")

        finally:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    # Default behavior: strict under/equal training FLOPs.
    # --include-more-flops: include only slightly above-baseline points within tolerance,
    #   and rank by absolute closeness to the baseline.
    # --allow-train-exceed: legacy unbounded above-baseline search.
    require_train_within = not args.allow_train_exceed
    prefer_under_target = not (args.include_more_flops or args.allow_train_exceed)

    ranked_all = rank_candidates(
        candidates,
        target_depth=baseline_depth,
        require_train_within=require_train_within,
        require_params_within=args.require_params_leq,
        prefer_under_target=prefer_under_target,
    )

    print_candidate_table(
        ranked_all,
        title="Best candidates across requested depths",
        top_k=args.top_k,
    )

    for depth in depths:
        depth_candidates = [c for c in candidates if c["depth"] == depth]
        ranked_depth = rank_candidates(
            depth_candidates,
            target_depth=depth,
            require_train_within=require_train_within,
            require_params_within=args.require_params_leq,
            prefer_under_target=prefer_under_target,
        )
        print_candidate_table(
            ranked_depth,
            title=f"Best candidates at fixed depth = {depth}",
            top_k=args.top_k,
        )

    if errors:
        print(f"\nEncountered {len(errors)} candidate errors. First 10:")
        for e in errors[:10]:
            print(f"  {e}")

    if args.save_json is not None:
        output = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "target_size": args.target_size,
            "baseline": baseline,
            "search_args": {
                "mlp_ratio": args.mlp_ratio,
                "depths": depths,
                "num_heads": candidate_heads,
                "embed_divisor": embed_divisor,
                "num_classes": num_classes,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "flops_mode": args.flops_mode,
                "train_multiplier": args.train_multiplier,
                "also_forward": args.also_forward,
                "embed_min": args.embed_min,
                "embed_max": args.embed_max,
                "embed_step": args.embed_step,
                "allow_train_exceed": args.allow_train_exceed,
                "include_more_flops": args.include_more_flops,
                "more_flops_pct": args.more_flops_pct,
                "more_params_pct": args.more_params_pct,
                "require_params_leq": args.require_params_leq,
            },
            "ranked_candidates": ranked_all,
            "all_candidates": candidates,
            "errors": errors,
        }
        atomic_save_json(output, args.save_json)
        print(f"\nSaved JSON: {args.save_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
