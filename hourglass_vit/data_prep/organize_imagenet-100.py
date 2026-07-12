#!/usr/bin/env python3
"""
Export a Hugging Face ImageNet100 dataset into OFDB/ImageFolder-style folders.

This is intended for:
  https://huggingface.co/datasets/ilee0022/ImageNet100

Important:
  - This HF dataset has label IDs 0..99 and a text field with class names.
  - It is not guaranteed to match the Kataoka/FractalDB supplementary synset list.
  - This script follows the HF dataset's own label/text mapping.

Default output:
  ImageNet-100-HF/
    train/
      000_bittern/
      001_conch/
      ...
    val/
      000_bittern/
      001_conch/
      ...
    imagenet100_hf_metadata.json
    classes.txt

For OFDB-style finetuning:
  data.trainset.root=./ImageNet-100-HF/train
  data.valset.root=./ImageNet-100-HF/val
  data.baseinfo.num_classes=100
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


def sanitize_class_name(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "unknown"
    return text[:max_len].strip("_")


def ensure_empty_or_allowed(path: Path, allow_existing: bool) -> None:
    if path.exists() and not allow_existing:
        if any(path.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {path}\n"
                f"Use --allow-existing to append/skip existing files, or choose a new --output-root."
            )
    path.mkdir(parents=True, exist_ok=True)


def new_size_stats() -> Dict[str, Any]:
    return {
        "checked": 0,
        "size_counts": {},
        "min_width": None,
        "max_width": None,
        "min_height": None,
        "max_height": None,
    }


def update_size_stats(stats: Dict[str, Any], width: int, height: int) -> None:
    stats["checked"] += 1
    key = f"{width}x{height}"
    stats["size_counts"][key] = stats["size_counts"].get(key, 0) + 1
    stats["min_width"] = width if stats["min_width"] is None else min(stats["min_width"], width)
    stats["max_width"] = width if stats["max_width"] is None else max(stats["max_width"], width)
    stats["min_height"] = height if stats["min_height"] is None else min(stats["min_height"], height)
    stats["max_height"] = height if stats["max_height"] is None else max(stats["max_height"], height)


def infer_available_splits(dataset_name: str, cache_dir: Optional[str]) -> List[str]:
    try:
        from datasets import get_dataset_split_names
    except Exception as e:
        raise RuntimeError("Please install Hugging Face datasets: pip install datasets") from e
    return list(get_dataset_split_names(dataset_name, cache_dir=cache_dir))


def load_hf_split(dataset_name: str, split: str, cache_dir: Optional[str]):
    from datasets import load_dataset
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def collect_label_text_map(ds) -> Dict[int, str]:
    """
    Build label_id -> class text from the HF split.

    The ilee0022/ImageNet100 viewer shows:
      image: Image
      label: int64
      text: string class name
    """
    label_to_text: Dict[int, str] = {}

    for ex in ds:
        if "label" not in ex:
            raise KeyError("HF example does not contain a 'label' field.")
        label = int(ex["label"])

        if "text" in ex and ex["text"] is not None:
            text = str(ex["text"])
        else:
            text = str(label)

        if label in label_to_text and label_to_text[label] != text:
            # Keep the first value but warn through metadata later.
            continue
        label_to_text[label] = text

    return dict(sorted(label_to_text.items(), key=lambda kv: kv[0]))


def make_folder_map(label_to_text: Dict[int, str]) -> Dict[int, str]:
    folder_map = {}
    used = set()
    for label, text in sorted(label_to_text.items()):
        base = f"{label:03d}_{sanitize_class_name(text)}"
        folder = base
        suffix = 2
        while folder in used:
            folder = f"{base}_{suffix}"
            suffix += 1
        folder_map[label] = folder
        used.add(folder)
    return folder_map


def print_split_summary(split_out: str, summary: Dict[str, Any], folder_map: Dict[int, str]) -> None:
    counts = summary["class_counts"]
    size_stats_by_label = summary["size_stats_by_label"]
    split_size_stats = summary["split_size_stats"]

    print(f"\nSplit '{split_out}' summary:")
    print(f"  {'Label':>5} {'Folder':<45} {'Images':>8} {'Checked':>8} {'Observed sizes':>24}")
    print("  " + "-" * 98)
    for label in sorted(folder_map):
        folder = folder_map[label]
        stats = size_stats_by_label[label]
        if stats["checked"] > 0:
            top_sizes = sorted(stats["size_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            size_text = ", ".join([f"{k}({v})" for k, v in top_sizes])
        else:
            size_text = "not checked"
        print(f"  {label:>5} {folder:<45} {counts[label]:>8} {stats['checked']:>8} {size_text:>24}")

    print("  " + "-" * 98)
    print(f"  {'TOTAL':>5} {'':<45} {summary['total_count']:>8} {split_size_stats['checked']:>8}")

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


def export_split(
    *,
    ds,
    output_root: Path,
    split_out: str,
    folder_map: Dict[int, str],
    image_format: str,
    jpeg_quality: int,
    overwrite: bool,
    dry_run: bool,
    limit_per_class: Optional[int],
    size_check_per_class: int,
) -> Dict[str, Any]:
    out_split_dir = output_root / split_out
    if not dry_run:
        out_split_dir.mkdir(parents=True, exist_ok=True)
        for label in folder_map:
            (out_split_dir / folder_map[label]).mkdir(parents=True, exist_ok=True)

    class_counts = {label: 0 for label in folder_map}
    size_stats_by_label = {label: new_size_stats() for label in folder_map}
    split_size_stats = new_size_stats()
    actions = defaultdict(int)

    fmt = image_format.lower()
    suffix = ".png" if fmt == "png" else ".jpg"

    for idx, ex in enumerate(ds):
        label = int(ex["label"])
        if label not in folder_map:
            raise ValueError(f"Encountered label {label}, but it is absent from the label map.")

        if limit_per_class is not None and class_counts[label] >= limit_per_class:
            continue

        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        if size_check_per_class != 0:
            should_check = size_check_per_class < 0 or size_stats_by_label[label]["checked"] < size_check_per_class
            if should_check:
                update_size_stats(size_stats_by_label[label], width, height)
                update_size_stats(split_size_stats, width, height)

        folder = folder_map[label]
        out_name = f"{folder}_{class_counts[label]:06d}{suffix}"
        out_path = out_split_dir / folder / out_name

        if dry_run:
            action = "planned"
        else:
            if out_path.exists() and not overwrite:
                action = "skipped_exists"
            else:
                if out_path.exists() and overwrite:
                    out_path.unlink()
                if fmt == "png":
                    img.save(out_path, format="PNG")
                else:
                    img.save(out_path, format="JPEG", quality=jpeg_quality)
                action = "saved"

        class_counts[label] += 1
        actions[action] += 1

    summary = {
        "class_counts": class_counts,
        "total_count": sum(class_counts.values()),
        "size_stats_by_label": size_stats_by_label,
        "split_size_stats": split_size_stats,
        "actions": dict(actions),
    }
    print_split_summary(split_out, summary, folder_map)
    print(f"Actions: {dict(actions)}")
    return summary


def write_metadata(
    *,
    output_root: Path,
    hf_dataset: str,
    hf_cache_dir: Optional[str],
    split_summaries: Dict[str, Dict[str, Any]],
    split_mapping: Dict[str, str],
    label_to_text: Dict[int, str],
    folder_map: Dict[int, str],
    args: argparse.Namespace,
) -> None:
    if args.dry_run:
        print("\nDry run: metadata not written.")
        return

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "ImageNet-100 from Hugging Face",
        "hf_dataset": hf_dataset,
        "hf_cache_dir": hf_cache_dir,
        "output_root": str(output_root),
        "num_classes": len(folder_map),
        "label_to_text": {str(k): v for k, v in label_to_text.items()},
        "label_to_folder": {str(k): v for k, v in folder_map.items()},
        "split_mapping_hf_to_output": split_mapping,
        "split_summaries": split_summaries,
        "split_counts": {
            split: {str(k): v for k, v in s["class_counts"].items()}
            for split, s in split_summaries.items()
        },
        "total_counts": {split: s["total_count"] for split, s in split_summaries.items()},
        "notes": [
            "This script follows the HF dataset label/text mapping.",
            "It does not assume that this HF dataset matches the Kataoka/FractalDB supplementary synset list.",
            "Output is ImageFolder-style for OFDB-style finetuning: train/class_folder/*.jpg and val/class_folder/*.jpg.",
            "Images are not resized here; use dataloader transform, e.g. timm.data.create_transform(input_size=224).",
        ],
    }

    if not args.no_metadata:
        with (output_root / "imagenet100_hf_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            f.write("\n")

    if not args.no_classes_file:
        with (output_root / "classes.txt").open("w", encoding="utf-8") as f:
            for label in sorted(folder_map):
                f.write(f"{label}\t{folder_map[label]}\t{label_to_text[label]}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HF ImageNet100 into OFDB/ImageFolder-style train/val folders."
    )
    parser.add_argument("--hf-dataset", default="ilee0022/ImageNet100")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="HF splits to export. Default: train validation.",
    )
    parser.add_argument(
        "--validation-output-name",
        default="val",
        help="Output folder name for HF validation split. Default: val.",
    )
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument(
        "--size-check-per-class",
        type=int,
        default=3,
        help="Use 0 to disable; use -1 to check every exported image.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-classes-file", action="store_true")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_empty_or_allowed(output_root, allow_existing=args.allow_existing or args.dry_run)

    print("=" * 96)
    print("Exporting Hugging Face ImageNet100 to ImageFolder")
    print(f"HF dataset:          {args.hf_dataset}")
    print(f"HF cache dir:        {args.hf_cache_dir}")
    print(f"Output root:         {output_root}")
    print(f"Requested HF splits: {args.splits}")
    print(f"Dry run:             {args.dry_run}")
    print(f"Size check:          {args.size_check_per_class} image(s) per class; -1 means all, 0 means disabled")
    print("=" * 96)

    available_splits = infer_available_splits(args.hf_dataset, args.hf_cache_dir)
    print(f"Available HF splits: {available_splits}")

    for s in args.splits:
        if s not in available_splits:
            raise ValueError(f"Requested split '{s}' is not available. Available: {available_splits}")

    # Build class mapping from train split if available; otherwise from the first requested split.
    mapping_split = "train" if "train" in available_splits else args.splits[0]
    print(f"\nBuilding label/text mapping from HF split: {mapping_split}")
    mapping_ds = load_hf_split(args.hf_dataset, mapping_split, args.hf_cache_dir)
    label_to_text = collect_label_text_map(mapping_ds)
    folder_map = make_folder_map(label_to_text)

    print(f"Detected {len(folder_map)} classes.")
    if len(folder_map) != 100:
        print(f"WARNING: Detected {len(folder_map)} classes, not 100.")

    print("First 10 classes:")
    for label in sorted(folder_map)[:10]:
        print(f"  {label:03d}: {folder_map[label]}  |  {label_to_text[label]}")

    split_summaries: Dict[str, Dict[str, Any]] = {}
    split_mapping: Dict[str, str] = {}

    for hf_split in args.splits:
        out_split = args.validation_output_name if hf_split == "validation" else hf_split
        split_mapping[hf_split] = out_split

        ds = load_hf_split(args.hf_dataset, hf_split, args.hf_cache_dir)
        summary = export_split(
            ds=ds,
            output_root=output_root,
            split_out=out_split,
            folder_map=folder_map,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            limit_per_class=args.limit_per_class,
            size_check_per_class=args.size_check_per_class,
        )
        split_summaries[out_split] = summary

    write_metadata(
        output_root=output_root,
        hf_dataset=args.hf_dataset,
        hf_cache_dir=args.hf_cache_dir,
        split_summaries=split_summaries,
        split_mapping=split_mapping,
        label_to_text=label_to_text,
        folder_map=folder_map,
        args=args,
    )

    total_all = sum(s["total_count"] for s in split_summaries.values())
    print("\nOverall summary:")
    for split, s in split_summaries.items():
        print(f"  {split:<8} images={s['total_count']:,}")
    print(f"  {'TOTAL':<8} images={total_all:,}")

    if args.dry_run:
        print("\nDry run completed. No output files were created.")
    else:
        print("\nDone. Output structure:")
        for split in split_summaries:
            print(f"  {output_root}/{split}/<class_folder>/*.jpg")
        if not args.no_metadata:
            print(f"Metadata written to: {output_root / 'imagenet100_hf_metadata.json'}")
        if not args.no_classes_file:
            print(f"Class order written to: {output_root / 'classes.txt'}")
        print("\nFor OFDB-style config:")
        if "train" in split_summaries:
            print(f"  data.trainset.root={output_root / 'train'}")
        if "val" in split_summaries:
            print(f"  data.valset.root={output_root / 'val'}")
        print(f"  data.baseinfo.num_classes={len(folder_map)}")


if __name__ == "__main__":
    main()
