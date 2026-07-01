import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _list_npz_files(paths):
    if isinstance(paths, str):
        paths = [paths]
    files = []
    for p in paths:
        for e in os.scandir(p):
            if e.is_file() and e.name.endswith(".npz"):
                files.append(e.path)
    files.sort()
    return files

def _rank_info():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
    except:
        pass
    return 0, 1

def _worker_sharded_indices(n_total, worker_id, num_workers, rank, world_size):
    step = num_workers * world_size
    offset = worker_id + rank * num_workers
    return range(offset, n_total, step)

def _rng_for_worker(base_seed, worker_id, rank):
    seed = (int(base_seed)
            + 0x85EBCA6B * (worker_id + 1)
            + 0xC2B2AE35 * (rank + 1)) & 0xFFFFFFFF
    return np.random.default_rng(seed)

class ImageNetHRIterDataset(IterableDataset):
    """
    單層 IterableDataset（無 epoch 依賴）：
      - 多資料夾 .npz
      - 正確 sharding（worker × DDP）
      - 初始化時可選擇 shuffle 一次
      - SR / Denoise 任務
      - 基礎增強（2 或 4 倍，固定循環）
    """
    def __init__(self,
                 folders,                     # str or list[str]
                 mode="super_resolution",     # "super_resolution" | "denoising"
                 down_scale=2.0,
                 noise_std=0.0,
                 unnormalize=True,
                 shuffle=True,                # 只在初始化時洗一次
                 seed=2024,
                 aug_num=None,                # None | 2 | 4
                 ):
        super().__init__()
        self.folders = folders
        self.mode = mode
        self.down_scale = float(down_scale)
        self.noise_std = float(noise_std)
        self.unnormalize = bool(unnormalize)
        self.seed = int(seed)
        self.aug_num = aug_num if aug_num in (None, 2, 4) else None

        # 準備完整檔案清單（主行程一次性）
        self._all_files = _list_npz_files(self.folders)
        if len(self._all_files) == 0:
            raise RuntimeError(f"No .npz files found under: {self.folders}")

        # 初始化時（最多）洗一次全域索引；之後固定
        self._indices = np.arange(len(self._all_files))
        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self._indices)

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
        elif aug_id == 1:
            return torch.flip(img, dims=[2])  # horizontal
        elif aug_id == 2:
            return torch.flip(img, dims=[1])  # vertical
        elif aug_id == 3:
            return torch.flip(torch.flip(img, dims=[2]), dims=[1])  # both
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

        # 固定的全域索引（已於 __init__ 洗過或未洗），這裡只做分片
        sharded_index_iter = _worker_sharded_indices(
            n_total=len(self._indices),
            worker_id=worker_id,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size
        )

        # 若有增強，為每個 worker/rank 決定起始偏移，使循環不完全同步
        if self.aug_num in (2, 4):
            rng_aug = _rng_for_worker(self.seed ^ 0xA5A5A5A5, worker_id, rank)
            aug_start = rng_aug.integers(0, self.aug_num)
        else:
            aug_start = 0

        k = 0
        for i in sharded_index_iter:
            npz_path = self._all_files[self._indices[i]]

            img = self._load_npz_to_tensor(npz_path)

            if self.aug_num in (2, 4):
                if self.aug_num == 2:
                    aug_id = [0, 3][(aug_start + k) % 2]   # 0 或 both
                else:
                    aug_id = (aug_start + k) % 4           # 0/h/v/both
                img = self._maybe_augment(img, aug_id)
                k += 1

            x, y = self._sr_or_denoise(img)
            yield x, y



dataset = ImageNetHRIterDataset(
    folders=[...],               # 你的 9 個 train_patch*
    mode="super_resolution",     # 或 "denoising"
    down_scale=2.0,
    noise_std=0.0,
    unnormalize=True,
    shuffle_each_epoch=False,    # 單 epoch 就不用每 epoch 重洗
    seed=2024,
    aug_num=4                    # 需要就用 2 或 4；不用就 None
)

def _worker_init_fn(_):
    torch.set_num_threads(1)     # 避免每個 worker 各自開太多 BLAS threads

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BS,
    num_workers=8,               # 視 CPU/SSD 調
    pin_memory=True,             # 預設是 False，要手動開
    prefetch_factor=4,           # 比預設 2 再多一點通常更順
    persistent_workers=True,     # 降低反覆啟停成本
    worker_init_fn=_worker_init_fn,
)


for x_cpu, y_cpu in loader: x = x_cpu.to("cuda", non_blocking=True) y = y_cpu.to("cuda", non_blocking=True) ...
