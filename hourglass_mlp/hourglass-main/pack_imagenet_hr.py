#!/usr/bin/env python3
"""
pack_imagenet_hr.py

將多個 ImageNet-HR .npz 檔打包成大 .npy shard。
每個 shard 會包含 shard_size 張圖，並生成同名 .json metadata。
會根據原始 patch 的 dtype 自動選擇 float16 / float32，不再強制轉型。

Usage:
    python3 pack_imagenet_hr.py --root ./data/ImageNet-HR --shard_size 100000
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List

# ---------------------------------------------------------
# 工具
# ---------------------------------------------------------

def _load_npz_patch(path: str) -> np.ndarray:
    """讀取 .npz 中的 patch（保持原始 dtype，不做 unnormalize）。"""
    with np.load(path, allow_pickle=False) as data:
        arr = data["patch"]
    # 通常是 float32 或 float16
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"Unsupported dtype {arr.dtype} in {path}")
    return arr  # (C,H,W)

def _scan_npz_dirs(dirs: List[str]):
    """逐層掃描資料夾，yield 所有 .npz 檔完整路徑（不建完整清單）。"""
    for d in dirs:
        if not os.path.isdir(d):
            continue
        with os.scandir(d) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".npz"):
                    yield e.path

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 寫 shard
# ---------------------------------------------------------

def _write_shard(out_dir: str, shard_idx: int, buf: np.memmap,
                 count: int, c: int, h: int, w: int, dtype: str):
    """將已填入 count 筆的 memmap buffer 寫成最終 shard 檔。"""
    if count == 0:
        return
    npy_path = os.path.join(out_dir, f"shard_{shard_idx:06d}.npy")
    meta_path = os.path.join(out_dir, f"shard_{shard_idx:06d}.json")

    out = np.lib.format.open_memmap(npy_path, mode="w+",
                                    dtype=buf.dtype, shape=(count, c, h, w))
    out[:] = buf[:count]
    del out  # flush to disk

    meta = {"count": int(count), "shape": [int(c), int(h), int(w)], "dtype": dtype}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"[write] {npy_path}  count={count}  shape=({c},{h},{w})  dtype={dtype}")

# ---------------------------------------------------------
# 打包主函式
# ---------------------------------------------------------

def pack_split(src_dirs: List[str], out_dir: str, shard_size: int = 100_000):
    """從多個資料夾讀取 .npz，依序打包成 shard。"""
    _ensure_dir(out_dir)

    # 掃第一張以確定形狀與 dtype
    first = None
    for p in _scan_npz_dirs(src_dirs):
        first = _load_npz_patch(p)
        break
    if first is None:
        print(f"[warn] no npz under {src_dirs}, skip")
        return
    C, H, W = map(int, first.shape)
    dtype = str(first.dtype)
    print(f"[info] detected shape: C={C}, H={H}, W={W}, dtype={dtype}")

    np_dtype = first.dtype
    shard_idx = 0
    processed = 0
    fill = 0

    def _new_memmap(max_count: int):
        tmp_path = os.path.join(out_dir, "__tmp_shard.npy")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        shape = (max_count, C, H, W)
        m = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np_dtype, shape=shape)
        return m, tmp_path

    buf, tmp_path = _new_memmap(shard_size)

    try:
        for i, npz_path in enumerate(_scan_npz_dirs(src_dirs), start=1):
            arr = _load_npz_patch(npz_path)
            if arr.shape != (C, H, W):
                raise RuntimeError(f"shape mismatch at {npz_path}: got {arr.shape}, expected {(C,H,W)}")
            if arr.dtype != np_dtype:
                # 若中途 dtype 不一致，轉成第一次的 dtype
                arr = arr.astype(np_dtype, copy=False)

            buf[fill] = arr
            fill += 1
            processed += 1

            if fill == shard_size:
                _write_shard(out_dir, shard_idx, buf, fill, C, H, W, str(np_dtype))
                del buf
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                shard_idx += 1
                buf, tmp_path = _new_memmap(shard_size)
                fill = 0

            if (processed % 10_000) == 0:
                print(f"[pack] processed {processed} files...")

    finally:
        _write_shard(out_dir, shard_idx, buf, fill, C, H, W, str(np_dtype))
        del buf
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"[done] total processed: {processed}")

# ---------------------------------------------------------
# 主程式入口
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data/ImageNet-HR",
                    help="root folder of ImageNet-HR")
    ap.add_argument("--shard_size", type=int, default=100_000,
                    help="images per shard (default=100000)")
    args = ap.parse_args()

    root = args.root
    train_dirs = [os.path.join(root, f"train_patch{i}") for i in range(1, 10)]
    val_dirs   = [os.path.join(root, "val")]
    test_dirs  = [os.path.join(root, "test")]

    out_train = os.path.join(root, "train_pack")
    out_val   = os.path.join(root, "val_pack")
    out_test  = os.path.join(root, "test_pack")

    print("== pack train ==")
    pack_split(train_dirs, out_train, args.shard_size)

    print("== pack val ==")
    pack_split(val_dirs, out_val, args.shard_size)

    print("== pack test ==")
    pack_split(test_dirs, out_test, args.shard_size)

if __name__ == "__main__":
    main()
