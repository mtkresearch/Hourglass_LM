#!/usr/bin/env python3
"""
Organize Places-30 (P30) from Places365 into an OFDB/ImageFolder-style dataset.

This version keeps the output root clean:
  - output-root contains only the selected Places-30 train/val folders
    plus optional metadata/classes files.
  - the full Places365 source/cache is stored outside output-root.

Default behavior:
  python3 organize_places-30.py --output-root ./Places-30 --download

This will use:
  Places365 source/cache: ./Places365_raw
  Places-30 output:       ./Places-30

Output format:
  Places-30/
    train/
      amphitheater/
      beauty_salon/
      ...
    val/
      amphitheater/
      beauty_salon/
      ...

For OFDB-style finetuning, point:
  train root -> ./Places-30/train
  val root   -> ./Places-30/val
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from torchvision.datasets import Places365


# Table 2, Categories in Places-30, from Kataoka et al. ACCV 2020 supplementary material.
# Keys are flat ImageFolder class names; values are Places365 category paths.
PLACES30: Dict[str, str] = {
    "amphitheater": "/a/amphitheater",
    "beauty_salon": "/b/beauty_salon",
    "boat_deck": "/b/boat_deck",
    "bowling_alley": "/b/bowling_alley",
    "butchers_shop": "/b/butchers_shop",
    "cliff": "/c/cliff",
    "creek": "/c/creek",
    "escalator_indoor": "/e/escalator/indoor",
    "flea_market_indoor": "/f/flea_market/indoor",
    "florist_shop_indoor": "/f/florist_shop/indoor",
    "food_court": "/f/food_court",
    "grotto": "/g/grotto",
    "harbor": "/h/harbor",
    "heliport": "/h/heliport",
    "industrial_area": "/i/industrial_area",
    "kennel_outdoor": "/k/kennel/outdoor",
    "manufactured_home": "/m/manufactured_home",
    "medina": "/m/medina",
    "motel": "/m/motel",
    "mountain": "/m/mountain",
    "park": "/p/park",
    "promenade": "/p/promenade",
    "restaurant_patio": "/r/restaurant_patio",
    "roof_garden": "/r/roof_garden",
    "server_room": "/s/server_room",
    "sky": "/s/sky",
    "skyscraper": "/s/skyscraper",
    "tundra": "/t/tundra",
    "valley": "/v/valley",
    "volcano": "/v/volcano",
}

SPLIT_MAP = {
    "train": "train-standard",
    "val": "val",
}


def normalize_category_name(name: str) -> str:
    """Normalize Places365 category strings for robust matching."""
    name = name.strip()
    if not name.startswith("/"):
        name = "/" + name
    return name.replace("\\", "/")


def get_class_indices(dataset: Places365) -> Dict[str, int]:
    """Return normalized Places365 class path -> index."""
    return {normalize_category_name(k): int(v) for k, v in dataset.class_to_idx.items()}


def ensure_empty_or_allowed(path: Path, allow_existing: bool) -> None:
    """Refuse to write into a non-empty output root unless explicitly allowed."""
    if path.exists() and not allow_existing:
        if any(path.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {path}\n"
                f"Use --allow-existing to append/skip existing files, or choose a new --output-root."
            )
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    """Create one output file and return an action string."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "skipped_exists"
        if dst.is_dir():
            raise IsADirectoryError(f"Destination is a directory, refusing to overwrite: {dst}")
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return "symlinked"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlinked"
    raise ValueError(f"Unknown mode: {mode}")


def unique_destination(dst_dir: Path, src_path: Path, src_root: Path) -> Path:
    """
    Build a stable destination filename.

    Most Places365 filenames are unique within a class. If a collision occurs,
    prefix the nested relative path.
    """
    dst = dst_dir / src_path.name
    if not dst.exists() and not dst.is_symlink():
        return dst

    try:
        rel = src_path.relative_to(src_root)
        safe = "__".join(rel.parts)
    except ValueError:
        safe = str(src_path).replace(os.sep, "__").replace(":", "")
    return dst_dir / safe


def new_size_stats() -> Dict[str, Any]:
    return {
        "checked": 0,
        "size_counts": {},
        "min_width": None,
        "max_width": None,
        "min_height": None,
        "max_height": None,
    }


def update_size_stats(size_stats: Dict[str, Any], image_path: Path) -> None:
    """Open one image and update compact width/height statistics."""
    with Image.open(image_path) as img:
        width, height = img.size

    size_stats["checked"] += 1
    key = f"{width}x{height}"
    size_stats["size_counts"][key] = size_stats["size_counts"].get(key, 0) + 1
    size_stats["min_width"] = width if size_stats["min_width"] is None else min(size_stats["min_width"], width)
    size_stats["max_width"] = width if size_stats["max_width"] is None else max(size_stats["max_width"], width)
    size_stats["min_height"] = height if size_stats["min_height"] is None else min(size_stats["min_height"], height)
    size_stats["max_height"] = height if size_stats["max_height"] is None else max(size_stats["max_height"], height)


def build_one_split(
    *,
    places365_root: Path,
    output_root: Path,
    output_split_name: str,
    places365_split_name: str,
    small: bool,
    download: bool,
    mode: str,
    overwrite: bool,
    limit_per_class: int | None,
    size_check_per_class: int,
    dry_run: bool,
) -> Dict[str, Any]:
    print(f"\nLoading Places365 split='{places365_split_name}', small={small}, download={download}")
    dataset = Places365(
        root=str(places365_root),
        split=places365_split_name,
        small=small,
        download=download,
        transform=None,
        target_transform=None,
    )

    class_to_idx = get_class_indices(dataset)

    selected_idx_to_name: Dict[int, str] = {}
    missing: List[Tuple[str, str]] = []
    for flat_name, category_path in PLACES30.items():
        key = normalize_category_name(category_path)
        if key not in class_to_idx:
            missing.append((flat_name, key))
        else:
            selected_idx_to_name[class_to_idx[key]] = flat_name

    if missing:
        available_hint = "\n".join(sorted(class_to_idx.keys())[:20])
        raise KeyError(
            "Some Places-30 categories were not found in torchvision Places365 metadata:\n"
            + "\n".join([f"  {flat}: {path}" for flat, path in missing])
            + "\n\nFirst available categories include:\n"
            + available_hint
        )

    output_split_dir = output_root / output_split_name
    if not dry_run:
        output_split_dir.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    actions = defaultdict(int)
    size_stats_by_class = {class_name: new_size_stats() for class_name in PLACES30}
    split_size_stats = new_size_stats()

    src_root = Path(dataset.root)

    for src_file, target in dataset.imgs:
        if target not in selected_idx_to_name:
            continue

        class_name = selected_idx_to_name[int(target)]
        if limit_per_class is not None and counts[class_name] >= limit_per_class:
            continue

        src_path = Path(src_file)
        if not src_path.exists():
            raise FileNotFoundError(f"Image file listed by Places365 does not exist: {src_path}")

        # Optional lightweight image-size inspection.
        # size_check_per_class=0 disables it; a negative value checks all selected images.
        if size_check_per_class != 0:
            already_checked = size_stats_by_class[class_name]["checked"]
            should_check = size_check_per_class < 0 or already_checked < size_check_per_class
            if should_check:
                update_size_stats(size_stats_by_class[class_name], src_path)
                update_size_stats(split_size_stats, src_path)

        dst_dir = output_split_dir / class_name
        dst_path = unique_destination(dst_dir, src_path, src_root)

        if dry_run:
            action = "planned"
        else:
            action = link_or_copy(src_path, dst_path, mode=mode, overwrite=overwrite)

        counts[class_name] += 1
        actions[action] += 1

    # Ensure all class directories exist even if a class unexpectedly has 0 images.
    if not dry_run:
        for flat_name in PLACES30:
            (output_split_dir / flat_name).mkdir(parents=True, exist_ok=True)

    print(f"Split '{output_split_name}' summary:")
    print(f"  {'Class':<24} {'Images':>8} {'Checked':>8} {'Observed sizes':>24}")
    print("  " + "-" * 68)
    for name in PLACES30:
        stats = size_stats_by_class[name]
        if stats["checked"] > 0:
            top_sizes = sorted(stats["size_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            size_text = ", ".join([f"{k}({v})" for k, v in top_sizes])
        else:
            size_text = "not checked"
        print(f"  {name:<24} {counts[name]:>8} {stats['checked']:>8} {size_text:>24}")

    total_count = sum(counts.values())
    print(f"  {'TOTAL':<24} {total_count:>8} {split_size_stats['checked']:>8}")
    if split_size_stats["checked"] > 0:
        top_sizes = sorted(split_size_stats["size_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        print("  Observed split image sizes:")
        for size_key, n in top_sizes:
            print(f"    {size_key:<12} {n}")
        print(
            "  Width range / height range among checked images: "
            f"{split_size_stats['min_width']}–{split_size_stats['max_width']} / "
            f"{split_size_stats['min_height']}–{split_size_stats['max_height']}"
        )
    print(f"Actions: {dict(actions)}")

    return {
        "class_counts": dict(counts),
        "total_count": total_count,
        "size_stats_by_class": size_stats_by_class,
        "split_size_stats": split_size_stats,
    }


def write_metadata(
    *,
    output_root: Path,
    places365_root: Path,
    small: bool,
    mode: str,
    split_summaries: Dict[str, Dict[str, Any]],
    dry_run: bool,
    write_metadata_file: bool,
    write_classes_file: bool,
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": "Places365",
        "source_root": str(places365_root),
        "small_256_images": small,
        "output_root": str(output_root),
        "mode": mode,
        "places30_categories": PLACES30,
        "split_summaries": split_summaries,
        "split_counts": {split: summary["class_counts"] for split, summary in split_summaries.items()},
        "total_counts": {split: summary["total_count"] for split, summary in split_summaries.items()},
        "notes": [
            "Places-30 category list follows Table 2 of Kataoka et al. ACCV 2020 supplementary material.",
            "Output is ImageFolder-style for OFDB-style fine-tuning: train/class_name/*.jpg and val/class_name/*.jpg.",
            "The full Places365 source/cache is intentionally kept outside output-root.",
            "If small_256_images is true, torchvision Places365 uses images resized to 256 x 256 pixels.",
        ],
    }

    if dry_run:
        print("\nDry run: metadata not written.")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    output_root.mkdir(parents=True, exist_ok=True)

    if write_metadata_file:
        with (output_root / "places30_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            f.write("\n")

    if write_classes_file:
        with (output_root / "classes.txt").open("w", encoding="utf-8") as f:
            for idx, class_name in enumerate(PLACES30.keys()):
                f.write(f"{idx}\t{class_name}\t{PLACES30[class_name]}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct Places-30 from Places365 and export only the selected "
            "ImageFolder-style train/val folders to output-root."
        )
    )
    parser.add_argument(
        "--places365-root",
        type=Path,
        default=None,
        help=(
            "Root directory used by torchvision.datasets.Places365. "
            "If omitted, defaults to <output-root parent>/Places365_raw, so output-root stays clean."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output Places-30 root directory. This will contain train/ and val/ only, plus optional metadata files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Output splits to build. 'train' maps to Places365 train-standard; 'val' maps to Places365 val.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help=(
            "How to materialize output files. "
            "symlink saves space but requires Places365 source/cache to remain available; "
            "copy creates a self-contained Places-30 output; hardlink saves space only on the same filesystem."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Ask torchvision to download missing Places365 devkit/images. This can be large.",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use high-resolution Places365 images instead of small 256x256 images. Default uses small=True.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow output-root to already exist; existing files are skipped unless --overwrite is set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output symlinks/files.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional debugging limit. Example: --limit-per-class 10 creates a tiny subset.",
    )
    parser.add_argument(
        "--size-check-per-class",
        type=int,
        default=3,
        help=(
            "How many selected images per class to open with PIL and inspect for stored image size. "
            "Use 0 to disable; use -1 to check every selected image, which can be slow."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and print counts; do not create symlinks/copies.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write places30_metadata.json. The terminal summary will still be printed.",
    )
    parser.add_argument(
        "--no-classes-file",
        action="store_true",
        help="Do not write classes.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    small = not args.large

    output_root = args.output_root.resolve()
    if args.places365_root is None:
        # Keep the Places365 source/cache outside output-root by default.
        places365_root = output_root.parent / "Places365_raw"
    else:
        places365_root = args.places365_root.resolve()

    # Guard against accidentally using the same folder for source and output.
    if places365_root == output_root:
        raise ValueError(
            "--places365-root cannot be the same as --output-root. "
            "The full Places365 source/cache would pollute the Places-30 output folder."
        )

    ensure_empty_or_allowed(output_root, allow_existing=args.allow_existing or args.dry_run)

    print("=" * 96)
    print("Building Places-30 from Places365")
    print(f"Places365 source/cache root: {places365_root}")
    print(f"Places-30 output root:       {output_root}")
    print(f"Splits:                      {args.splits}")
    print(f"Image variant:               {'small 256x256' if small else 'large/high-resolution'}")
    print(f"Mode:                        {args.mode}")
    print(f"Download:                    {args.download}")
    print(f"Dry run:                     {args.dry_run}")
    print(f"Size check:                  {args.size_check_per_class} image(s) per class; -1 means all, 0 means disabled")
    print(f"Write metadata:              {not args.no_metadata}")
    print(f"Write classes file:          {not args.no_classes_file}")
    print("=" * 96)

    split_summaries: Dict[str, Dict[str, Any]] = {}
    for out_split in args.splits:
        places_split = SPLIT_MAP[out_split]
        summary = build_one_split(
            places365_root=places365_root,
            output_root=output_root,
            output_split_name=out_split,
            places365_split_name=places_split,
            small=small,
            download=args.download,
            mode=args.mode,
            overwrite=args.overwrite,
            limit_per_class=args.limit_per_class,
            size_check_per_class=args.size_check_per_class,
            dry_run=args.dry_run,
        )
        split_summaries[out_split] = summary

    write_metadata(
        output_root=output_root,
        places365_root=places365_root,
        small=small,
        mode=args.mode,
        split_summaries=split_summaries,
        dry_run=args.dry_run,
        write_metadata_file=not args.no_metadata,
        write_classes_file=not args.no_classes_file,
    )

    if args.dry_run:
        print("\nDry run completed. No output files were created.")
    else:
        print("\nDone. Clean Places-30 output structure:")
        for split in args.splits:
            print(f"  {output_root}/{split}/<class_name>/*.jpg")
        if not args.no_metadata:
            print(f"Metadata written to: {output_root / 'places30_metadata.json'}")
        if not args.no_classes_file:
            print(f"Class order written to: {output_root / 'classes.txt'}")
        print("\nFor your OFDB-style config:")
        print(f"  data.trainset.root={output_root / 'train'}")
        print(f"  data.valset.root={output_root / 'val'}")
        print("  data.baseinfo.num_classes=30")


if __name__ == "__main__":
    main()
