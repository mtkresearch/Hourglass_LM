import os
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import random
from collections import deque
from torch.utils.data import IterableDataset, get_worker_info

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _rank_info():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
    except:
        pass
    return 0, 1

def _stable_hash_mod(s: str, mod: int) -> int:
    """對字串做穩定 hash（跨進程/重啟一致），回傳 0..mod-1。"""
    if mod <= 1:
        return 0
    h = hashlib.md5(s.encode('utf-8')).digest()
    v = int.from_bytes(h[:8], "little", signed=False)  # 64-bit
    return v % mod

class ImageNetHRStreamDataset(IterableDataset):
    """
    針對超大量 .npz 設計的單層 IterableDataset：
      - 不建立全域清單，直接 scandir() 流式列舉
      - 正確 sharding（worker × DDP）：資料夾分片優先、否則檔名哈希分片
      - 可選 block_shuffle（近似隨機，無需總長度）
      - SR / Denoise 任務、簡單翻轉增強
    """
    def __init__(self,
                 folders,                     # str 或 list[str]，如 [".../train_patch1", ... "train_patch9"]
                 mode="super_resolution",     # "super_resolution" | "denoising"
                 down_scale=2.0,
                 noise_std=0.0,
                 unnormalize=True,
                 do_shuffle=False,            # 是否啟用 block-shuffle（非全域 shuffle）
                 block_shuffle_size=8192,     # shuffle 緩衝大小（依 RAM/速度調整；0/1 表示不啟用）
                 seed=2024,
                 aug_num=None,                # None | 2 | 4
                 ):
        super().__init__()
        if isinstance(folders, str):
            folders = [folders]
        self.folders = list(folders)
        self.mode = mode
        self.down_scale = float(down_scale)
        self.noise_std = float(noise_std)
        self.unnormalize = bool(unnormalize)
        self.seed = int(seed)
        self.aug_num = aug_num if aug_num in (None, 2, 4) else None

        # block-shuffle 只做局部緩衝打亂，不需要總長度
        self.do_shuffle = bool(do_shuffle and block_shuffle_size and block_shuffle_size > 1)
        self.block_shuffle_size = int(block_shuffle_size)

        # 檢查資料夾存在
        missing = [p for p in self.folders if not os.path.isdir(p)]
        if missing:
            raise FileNotFoundError(f"Folders not found: {missing}")

    def _load_npz_to_tensor(self, npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            patch = data["patch"]
        if patch.dtype != np.float32:
            patch = patch.astype(np.float32, copy=False)
        if self.unnormalize:
            patch = patch * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
            np.clip(patch, 0.0, 1.0, out=patch)
        return torch.from_numpy(patch)  # (C,H,W), float32

    @torch.inference_mode()
    def _maybe_augment(self, img: torch.Tensor, aug_id: int):
        if aug_id == 0:
            return img
        elif aug_id == 1:  # horizontal
            return torch.flip(img, dims=[2])
        elif aug_id == 2:  # vertical
            return torch.flip(img, dims=[1])
        elif aug_id == 3:  # both
            return torch.flip(torch.flip(img, dims=[2]), dims=[1])
        return img

    @torch.inference_mode()
    def _sr_or_denoise(self, img: torch.Tensor):
        if self.mode == "super_resolution":
            C, H, W = img.shape
            tH = int(H // self.down_scale)
            tW = int(W // self.down_scale)
            low = F.interpolate(img.unsqueeze(0), size=(tH, tW), mode="bicubic",
                                align_corners=False, antialias=False).squeeze(0)
            return low.view(-1), img.view(-1)
        elif self.mode == "denoising":
            noisy = torch.clamp(img + torch.randn_like(img) * self.noise_std, 0.0, 1.0)
            return noisy.view(-1), img.view(-1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __iter__(self):
        worker = get_worker_info()
        rank, world_size = _rank_info()

        if worker is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker.num_workers
            worker_id = worker.id

        # 全域 shard 設定
        total_shards = world_size * num_workers
        my_shard_id = worker_id + rank * num_workers

        # --- 決定本 shard 要掃哪些資料夾 ---
        # 若資料夾數 >= shard 數：用「資料夾分片」（可避免每個 worker 重掃所有資料夾）
        # 否則：每個 shard 都掃全部資料夾，但在「檔名層級」再哈希過濾
        if len(self.folders) >= total_shards and total_shards > 1:
            assigned_folders = [p for idx, p in enumerate(self.folders) if (idx % total_shards) == my_shard_id]
            hash_filter_mod = 1  # 不再需要檔名哈希分片
        else:
            assigned_folders = self.folders[:]   # 掃全部
            hash_filter_mod = total_shards       # 用檔名哈希分片

        # --- 增強循環的起點（避免所有 worker 同步）---
        rng_local = random.Random((self.seed ^ 0xA5A5A5A5) + my_shard_id)
        if self.aug_num in (2, 4):
            aug_cursor = rng_local.randrange(self.aug_num)
        else:
            aug_cursor = 0

        # --- block-shuffle 緩衝 ---
        buf = deque()
        rng_shuffle = random.Random(self.seed + 12345 + my_shard_id)

        def maybe_yield_from_buffer(force_drain=False):
            # 從緩衝區隨機彈出（近似洗牌）
            while buf and (force_drain or (self.do_shuffle and len(buf) >= self.block_shuffle_size)):
                j = rng_shuffle.randrange(len(buf)) if self.do_shuffle else 0
                npz_path = buf[j]
                # 把選中的移到左端再 popleft（O(1)）
                buf.rotate(-j)
                chosen = buf.popleft()
                buf.rotate(j)
                yield chosen

        # --- 主掃描回圈（完全流式）---
        for folder in assigned_folders:
            # 不遞迴：若需要遞迴，改成 os.walk
            with os.scandir(folder) as it:
                for e in it:
                    if not (e.is_file() and e.name.endswith(".npz")):
                        continue
                    # 若使用檔名哈希分片，過濾非本 shard 的檔案
                    if hash_filter_mod > 1:
                        if _stable_hash_mod(e.name, hash_filter_mod) != my_shard_id:
                            continue

                    # 先把路徑丟進緩衝
                    if self.do_shuffle:
                        buf.append(e.path)
                        # 當緩衝到達設定大小，隨機吐一批出去
                        for out_path in maybe_yield_from_buffer(force_drain=False):
                            img = self._load_npz_to_tensor(out_path)
                            if self.aug_num in (2, 4):
                                if self.aug_num == 2:
                                    aug_id = [0, 3][aug_cursor % 2]
                                else:
                                    aug_id = aug_cursor % 4
                                img = self._maybe_augment(img, aug_id)
                                aug_cursor += 1
                            x, y = self._sr_or_denoise(img)
                            yield x, y
                    else:
                        # 直接即時處理，不進緩衝
                        img = self._load_npz_to_tensor(e.path)
                        if self.aug_num in (2, 4):
                            if self.aug_num == 2:
                                aug_id = [0, 3][aug_cursor % 2]
                            else:
                                aug_id = aug_cursor % 4
                            img = self._maybe_augment(img, aug_id)
                            aug_cursor += 1
                        x, y = self._sr_or_denoise(img)
                        yield x, y

        # 掃描完後把緩衝吐光
        if self.do_shuffle:
            for out_path in maybe_yield_from_buffer(force_drain=True):
                img = self._load_npz_to_tensor(out_path)
                if self.aug_num in (2, 4):
                    if self.aug_num == 2:
                        aug_id = [0, 3][aug_cursor % 2]
                    else:
                        aug_id = aug_cursor % 4
                    img = self._maybe_augment(img, aug_id)
                    aug_cursor += 1
                x, y = self._sr_or_denoise(img)
                yield x, y
