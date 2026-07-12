#!/usr/bin/env python3
"""
Convert Pascal VOC 2012 detection annotations into an object-crop classification
dataset in ImageFolder style.

This follows the FractalDB/OFDB-style interpretation mentioned by the author:
  "the pascal voc classification is based on the detection. The bounding boxes
   are simply cropped as input images and used the paired label."

Output format:
  VOC12-crops/
    train/
      aeroplane/
      bicycle/
      ...
      tvmonitor/
    val/
      aeroplane/
      bicycle/
      ...
      tvmonitor/
    voc12_crop_metadata.json
    classes.txt

Default behavior:
  - Uses VOC2012 train and val image sets.
  - Excludes objects marked difficult=1, matching VOC evaluation object counts.
  - Crops each bounding box and saves it as a single-label image.
  - Does not resize crops; your dataloader transform should resize/crop to 224.

Example:
  python3 organize_voc12_crops.py \
    --voc-root ./VOC_raw \
    --output-root ./VOC12-crops \
    --download

For OFDB-style config:
  data.trainset.root=./VOC12-crops/train
  data.valset.root=./VOC12-crops/val
  data.baseinfo.num_classes=20
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

VOC_CLASSES: List[str] = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def ensure_empty_or_allowed(path: Path, allow_existing: bool) -> None:
    if path.exists() and not allow_existing:
        if any(path.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {path}\n"
                f"Use --allow-existing to append/skip existing files, or choose a new --output-root."
            )
    path.mkdir(parents=True, exist_ok=True)


def maybe_download_voc2012(voc_root: Path, download: bool) -> None:
    """Use torchvision only for downloading/extracting VOC2012 if requested."""
    if not download:
        return

    try:
        from torchvision.datasets import VOCDetection
    except Exception as e:
        raise RuntimeError(
            "torchvision is required for --download. Install torchvision or download VOC2012 manually."
        ) from e

    print(f"Downloading/checking VOC2012 via torchvision under: {voc_root}")
    # One call is enough; VOC2012 train/val live in the same trainval archive.
    _ = VOCDetection(
        root=str(voc_root),
        year="2012",
        image_set="train",
        download=True,
        transform=None,
        target_transform=None,
    )


def find_voc2012_dir(voc_root: Path) -> Path:
    """Accept either root/VOCdevkit/VOC2012, root/VOC2012, or root itself."""
    candidates = [
        voc_root / "VOCdevkit" / "VOC2012",
        voc_root / "VOC2012",
        voc_root,
    ]
    for c in candidates:
        if (
            (c / "JPEGImages").is_dir()
            and (c / "Annotations").is_dir()
            and (c / "ImageSets" / "Main").is_dir()
        ):
            return c

    raise FileNotFoundError(
        "Cannot find VOC2012 directory. Expected one of:\n"
        + "\n".join(str(c) for c in candidates)
        + "\nRequired subfolders: JPEGImages/, Annotations/, ImageSets/Main/."
    )


def read_split_ids(voc_dir: Path, split: str) -> List[str]:
    split_file = voc_dir / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Cannot find split file: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_xml_annotation(xml_path: Path) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> node in {xml_path}")

    image_size = {
        "width": int(size_node.findtext("width")),
        "height": int(size_node.findtext("height")),
        "depth": int(size_node.findtext("depth", default="3")),
    }

    objects: List[Dict[str, Any]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue
        name = name.strip()

        difficult = int(obj.findtext("difficult", default="0"))
        truncated = int(obj.findtext("truncated", default="0"))
        occluded = int(obj.findtext("occluded", default="0") or 0)

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        # VOC bboxes are 1-based inclusive. For PIL crop, use 0-based left/top
        # and right/bottom as exclusive. The common conversion is xmin-1, ymin-1,
        # xmax, ymax.
        xmin = int(float(bnd.findtext("xmin")))
        ymin = int(float(bnd.findtext("ymin")))
        xmax = int(float(bnd.findtext("xmax")))
        ymax = int(float(bnd.findtext("ymax")))

        objects.append(
            {
                "name": name,
                "difficult": difficult,
                "truncated": truncated,
                "occluded": occluded,
                "bbox_voc": [xmin, ymin, xmax, ymax],
            }
        )

    return image_size, objects


def voc_bbox_to_pil_box(
    bbox_voc: List[int],
    image_width: int,
    image_height: int,
    expand: float,
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox_voc

    left = max(0, xmin - 1)
    top = max(0, ymin - 1)
    right = min(image_width, xmax)
    bottom = min(image_height, ymax)

    if expand != 1.0:
        w = right - left
        h = bottom - top
        cx = left + w / 2.0
        cy = top + h / 2.0
        new_w = w * expand
        new_h = h * expand
        left = max(0, int(round(cx - new_w / 2.0)))
        top = max(0, int(round(cy - new_h / 2.0)))
        right = min(image_width, int(round(cx + new_w / 2.0)))
        bottom = min(image_height, int(round(cy + new_h / 2.0)))

    return left, top, right, bottom


def new_size_stats() -> Dict[str, Any]:
    return {
        "count": 0,
        "min_width": None,
        "max_width": None,
        "min_height": None,
        "max_height": None,
        "size_counts": {},
    }


def update_size_stats(stats: Dict[str, Any], width: int, height: int) -> None:
    stats["count"] += 1
    stats["min_width"] = width if stats["min_width"] is None else min(stats["min_width"], width)
    stats["max_width"] = width if stats["max_width"] is None else max(stats["max_width"], width)
    stats["min_height"] = height if stats["min_height"] is None else min(stats["min_height"], height)
    stats["max_height"] = height if stats["max_height"] is None else max(stats["max_height"], height)
    key = f"{width}x{height}"
    stats["size_counts"][key] = stats["size_counts"].get(key, 0) + 1


def print_split_summary(split: str, summary: Dict[str, Any]) -> None:
    counts = summary["class_counts"]
    crop_stats_by_class = summary["crop_size_stats_by_class"]

    print(f"\nSplit '{split}' summary:")
    print(f"  {'Class':<16} {'Crops':>8} {'Width range':>17} {'Height range':>17} {'Top sizes':>28}")
    print("  " + "-" * 92)
    for cls in VOC_CLASSES:
        stats = crop_stats_by_class[cls]
        if stats["count"] > 0:
            w_range = f"{stats['min_width']}–{stats['max_width']}"
            h_range = f"{stats['min_height']}–{stats['max_height']}"
            top_sizes = sorted(stats["size_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            size_text = ", ".join(f"{k}({v})" for k, v in top_sizes)
        else:
            w_range = "-"
            h_range = "-"
            size_text = "-"
        print(f"  {cls:<16} {counts[cls]:>8} {w_range:>17} {h_range:>17} {size_text:>28}")

    print("  " + "-" * 92)
    print(f"  {'TOTAL':<16} {summary['total_crops']:>8}")
    print(f"  Source images used: {summary['source_images']}")
    print(f"  Difficult objects skipped: {summary['skipped_difficult']}")
    print(f"  Invalid boxes skipped:      {summary['skipped_invalid_bbox']}")
    print(f"  Existing files skipped:     {summary['skipped_existing']}")
    print(f"  Crops written/planned:      {summary['written_or_planned']}")


def process_split(
    *,
    voc_dir: Path,
    output_root: Path,
    split: str,
    include_difficult: bool,
    expand_bbox: float,
    image_format: str,
    jpeg_quality: int,
    allow_existing: bool,
    overwrite: bool,
    dry_run: bool,
    limit_per_class: int | None,
) -> Dict[str, Any]:
    ids = read_split_ids(voc_dir, split)

    out_split_dir = output_root / split
    if not dry_run:
        for cls in VOC_CLASSES:
            (out_split_dir / cls).mkdir(parents=True, exist_ok=True)

    class_counts = {cls: 0 for cls in VOC_CLASSES}
    crop_size_stats_by_class = {cls: new_size_stats() for cls in VOC_CLASSES}
    split_crop_size_stats = new_size_stats()

    source_images_with_at_least_one_crop = set()
    skipped_difficult = 0
    skipped_invalid_bbox = 0
    skipped_existing = 0
    written_or_planned = 0

    fmt = image_format.lower()
    suffix = ".png" if fmt == "png" else ".jpg"

    for image_id in ids:
        xml_path = voc_dir / "Annotations" / f"{image_id}.xml"
        img_path = voc_dir / "JPEGImages" / f"{image_id}.jpg"

        if not xml_path.exists():
            raise FileNotFoundError(f"Missing annotation: {xml_path}")
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        image_size, objects = parse_xml_annotation(xml_path)
        width = image_size["width"]
        height = image_size["height"]

        # Open lazily only if this image has at least one selected crop.
        img = None
        try:
            for obj_idx, obj in enumerate(objects, start=1):
                cls = obj["name"]
                if cls not in VOC_CLASSES:
                    continue

                if obj["difficult"] == 1 and not include_difficult:
                    skipped_difficult += 1
                    continue

                if limit_per_class is not None and class_counts[cls] >= limit_per_class:
                    continue

                left, top, right, bottom = voc_bbox_to_pil_box(
                    obj["bbox_voc"],
                    image_width=width,
                    image_height=height,
                    expand=expand_bbox,
                )

                crop_w = right - left
                crop_h = bottom - top
                if crop_w <= 0 or crop_h <= 0:
                    skipped_invalid_bbox += 1
                    continue

                out_name = (
                    f"{image_id}_obj{obj_idx:02d}_"
                    f"{cls}_x{left}_y{top}_w{crop_w}_h{crop_h}{suffix}"
                )
                out_path = out_split_dir / cls / out_name

                if out_path.exists() and not overwrite:
                    skipped_existing += 1
                    class_counts[cls] += 1
                    update_size_stats(crop_size_stats_by_class[cls], crop_w, crop_h)
                    update_size_stats(split_crop_size_stats, crop_w, crop_h)
                    source_images_with_at_least_one_crop.add(image_id)
                    continue

                if not dry_run:
                    if img is None:
                        img = Image.open(img_path).convert("RGB")

                    crop = img.crop((left, top, right, bottom))
                    if out_path.exists() and overwrite:
                        out_path.unlink()

                    if fmt == "png":
                        crop.save(out_path, format="PNG")
                    else:
                        crop.save(out_path, format="JPEG", quality=jpeg_quality)

                class_counts[cls] += 1
                written_or_planned += 1
                update_size_stats(crop_size_stats_by_class[cls], crop_w, crop_h)
                update_size_stats(split_crop_size_stats, crop_w, crop_h)
                source_images_with_at_least_one_crop.add(image_id)

        finally:
            if img is not None:
                img.close()

    total_crops = sum(class_counts.values())
    summary = {
        "class_counts": class_counts,
        "total_crops": total_crops,
        "source_images": len(source_images_with_at_least_one_crop),
        "skipped_difficult": skipped_difficult,
        "skipped_invalid_bbox": skipped_invalid_bbox,
        "skipped_existing": skipped_existing,
        "written_or_planned": written_or_planned,
        "crop_size_stats_by_class": crop_size_stats_by_class,
        "split_crop_size_stats": split_crop_size_stats,
        "num_image_ids_in_split": len(ids),
    }
    print_split_summary(split, summary)
    return summary


def write_metadata(
    *,
    output_root: Path,
    voc_root: Path,
    voc_dir: Path,
    split_summaries: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if args.dry_run:
        print("\nDry run: metadata not written.")
        return

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": "PASCAL VOC 2012",
        "source_root": str(voc_root),
        "voc2012_dir": str(voc_dir),
        "output_root": str(output_root),
        "task_conversion": (
            "VOC2012 detection annotations converted to object-crop single-label classification. "
            "Each bounding box crop is paired with its object class label."
        ),
        "include_difficult": args.include_difficult,
        "exclude_difficult": not args.include_difficult,
        "expand_bbox": args.expand_bbox,
        "image_format": args.image_format,
        "jpeg_quality": args.jpeg_quality,
        "classes": VOC_CLASSES,
        "split_summaries": split_summaries,
        "split_counts": {split: s["class_counts"] for split, s in split_summaries.items()},
        "total_counts": {split: s["total_crops"] for split, s in split_summaries.items()},
        "notes": [
            "Default excludes objects marked difficult=1, matching VOC evaluation object counts.",
            "Crops are not resized here. Use your dataloader transform, e.g. timm.data.create_transform(input_size=224).",
            "This output is ImageFolder-style for OFDB-style finetuning: train/class_name/*.jpg and val/class_name/*.jpg.",
        ],
    }

    with (output_root / "voc12_crop_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if not args.no_classes_file:
        with (output_root / "classes.txt").open("w", encoding="utf-8") as f:
            for idx, cls in enumerate(VOC_CLASSES):
                f.write(f"{idx}\t{cls}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Pascal VOC 2012 detection bounding boxes into an "
            "ImageFolder-style object-crop classification dataset."
        )
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("./VOC_raw"),
        help=(
            "Root containing VOCdevkit/VOC2012, or where torchvision should download it. "
            "Default: ./VOC_raw"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for object-crop classification dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "trainval"],
        default=["train", "val"],
        help="VOC2012 image sets to process. Default: train val.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download VOC2012 trainval via torchvision if missing.",
    )
    parser.add_argument(
        "--include-difficult",
        action="store_true",
        help=(
            "Include objects marked difficult=1. Default is to exclude them, "
            "matching VOC evaluation object counts."
        ),
    )
    parser.add_argument(
        "--expand-bbox",
        type=float,
        default=1.0,
        help="Expand each bbox around its center before cropping. 1.0 means no expansion.",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output crop image format. Default: jpg.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality if --image-format jpg. Default: 95.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow output-root to already exist; existing files are skipped unless --overwrite is set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing crop files.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Debug option. Example: --limit-per-class 10.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse annotations and print expected crop counts, but do not write crop files.",
    )
    parser.add_argument(
        "--no-classes-file",
        action="store_true",
        help="Do not write classes.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    voc_root = args.voc_root.resolve()
    output_root = args.output_root.resolve()

    maybe_download_voc2012(voc_root, args.download)
    voc_dir = find_voc2012_dir(voc_root)

    ensure_empty_or_allowed(output_root, allow_existing=args.allow_existing or args.dry_run)

    print("=" * 96)
    print("Converting Pascal VOC 2012 detection annotations to object-crop classification")
    print(f"VOC root:          {voc_root}")
    print(f"VOC2012 dir:       {voc_dir}")
    print(f"Output root:       {output_root}")
    print(f"Splits:            {args.splits}")
    print(f"Include difficult: {args.include_difficult}")
    print(f"Expand bbox:       {args.expand_bbox}")
    print(f"Image format:      {args.image_format}")
    print(f"Dry run:           {args.dry_run}")
    print("=" * 96)

    split_summaries: Dict[str, Dict[str, Any]] = {}
    for split in args.splits:
        summary = process_split(
            voc_dir=voc_dir,
            output_root=output_root,
            split=split,
            include_difficult=args.include_difficult,
            expand_bbox=args.expand_bbox,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            allow_existing=args.allow_existing,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            limit_per_class=args.limit_per_class,
        )
        split_summaries[split] = summary

    write_metadata(
        output_root=output_root,
        voc_root=voc_root,
        voc_dir=voc_dir,
        split_summaries=split_summaries,
        args=args,
    )

    if args.dry_run:
        print("\nDry run completed. No crops were written.")
    else:
        print("\nDone. Output structure:")
        for split in args.splits:
            print(f"  {output_root}/{split}/<class_name>/*.jpg")
        print(f"Metadata written to: {output_root / 'voc12_crop_metadata.json'}")
        if not args.no_classes_file:
            print(f"Class order written to: {output_root / 'classes.txt'}")
        print("\nFor OFDB-style config:")
        if "train" in args.splits:
            print(f"  data.trainset.root={output_root / 'train'}")
        if "val" in args.splits:
            print(f"  data.valset.root={output_root / 'val'}")
        print("  data.baseinfo.num_classes=20")


if __name__ == "__main__":
    main()
