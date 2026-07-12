# Hourglass ViT

Code for **Hourglass ViT**.

## Table of Contents

- [Code Structure](#code-structure)
- [Environment Setup](#environment-setup)
- [Datasets Preparation](#datasets-preparation)
- [Training](#training)
  - [Training Hyperparameters](#training-hyperparameters)
- [Training FLOPs Measurement](#training-flops-measurement)
- [Inference Latency Calculation](#inference-latency-calculation)
- [Acknowledgements](#acknowledgements)

## Code Structure

```
hourglass_vit/
├── main.py                  # training entry point (hydra)
├── data.py                  # dataloaders (ImageFolder + timm transforms)
├── logger.py                # wandb + console logging
├── utils.py                 # distributed init (torchrun)
├── configs/                 # hydra configs (main / model=vit / data=colorimagefolder)
├── scripts/
│   └── train.sh             # unified training launcher
├── profiling/
│   ├── profile_flops.py     # forward / training FLOPs tables
│   ├── inference_latency.py # latency benchmark tables
│   └── search_config_by_train_flops.py
├── data_prep/               # dataset download & preprocessing (see its README)
└── data/                    # datasets live here by default (gitignored)
```

## Environment Setup

```bash
conda create -n hourglass-vit python=3.10 -y
conda activate hourglass-vit

# Install PyTorch version that suits your device
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Training logs to [Weights & Biases](https://wandb.ai): run `wandb login` once, or set
`WANDB_MODE=offline` to skip it.

## Datasets Preparation

See [`data_prep/README.md`](data_prep/README.md) for the download / preprocessing steps
of every dataset. After preparation they live under `./data/` (or set `DATA_ROOT`).

| `DATASET=` | Dataset | #Classes | Train / Val imgs |
|---|---|---:|---:|
| `cars` | Stanford Cars | 196 | 8,144 / 8,041 |
| `cifar10` | CIFAR-10 | 10 | 50,000 / 10,000 |
| `cifar100` | CIFAR-100 | 100 | 50,000 / 10,000 |
| `flowers` | Oxford Flowers-102 | 102 | 6,149 / 1,020 |
| `voc12` | Pascal VOC-12 (object crops) | 20 | 13,609 / 13,841 |
| `imagenet100` | ImageNet-100 | 100 | 117,000 / 13,000 |
| `places30` | Places-30 | 30 | 149,254 / 3,000 |

## Training

```bash
chmod +x scripts/*.sh

# Conventional ViT-Tiny (mlp_ratio = 4.0) on Stanford Cars, 2 GPUs
DATASET=cars CUDA_VISIBLE_DEVICES=0,1 ./scripts/train.sh

# Hourglass ViT-Tiny (mlp_ratio = 0.5, d_z = 288, L = 11) on Stanford Cars
DATASET=cars MLP_RATIO=0.5 EMBED_DIM=288 DEPTH=11 \
  CUDA_VISIBLE_DEVICES=0,1 ./scripts/train.sh

# Save the terminal log to txt file:
DATASET=cars CUDA_VISIBLE_DEVICES=0,1 \
  ./scripts/train.sh 2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log
```

Options:

| Variable | Meaning | Default |
|---|---|---|
| `DATASET` | `cars` \| `cifar10` \| `cifar100` \| `flowers` \| `voc12` \| `imagenet100` \| `places30` | required |
| `MODEL` | ViT size: `tiny` \| `small` \| `base` \| `large` | `tiny` |
| `MLP_RATIO` | `r = d_h / d_z` | 4.0 |
| `EMBED_DIM` | token dimension `d_z` | per MODEL: 192/384/768/1024 |
| `DEPTH` | number of layers `L` | per MODEL: 12/12/12/24 |
| `NUM_HEADS` | attention heads; `d_z` must be divisible by it | per MODEL: 3/6/12/16 |
| `LR` | base LR; the real LR is `LR × TOTAL_BS / 512` (DeiT linear scaling) | 3e-2 |
| `TOTAL_BS` | total batch size across GPUs | 768 (384 for `large`) |
| `EPOCHS` | training epochs | 1000 |
| `WARMUP_EPOCHS` | warmup epochs | 10 |
| `DATA_ROOT` | dataset parent dir | `./data` |
| `NUM_WORKERS` | dataloader workers per GPU process | 4 (20 for imagenet100/places30) |
| `NGPUS` | number of GPUs | inferred from `CUDA_VISIBLE_DEVICES` |
| `WANDB_PROJECT` | wandb project name | `hourglass-vit` |

Checkpoints and logs go to `./output/<experiment-name>/`.

### Training Hyperparameters

Already set in `scripts/train.sh` and `configs/data/colorimagefolder.yaml`

- Shared across all model sizes:

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD (momentum 0.9, nesterov) |
| Weight decay | 1e-4 |
| LR schedule | cosine, 10 warmup epochs |
| Epochs | 1000 |
| Precision | AMP (mixed precision) |
| Input size | 224 × 224 |
| Augmentation | RandAugment (rand-m9-mstd0.5-inc1), Mixup 0.8, CutMix 1.0, Random Erasing 0.25, label smoothing 0.1 |

- Different across model sizes:

| | Tiny | Small | Base | Large |
|---|---:|---:|---:|---:|
| Total batch size | 768 | 768 | 768 | 384 |
| Base LR | 3e-2 | 3e-2 | 1e-2 – 3e-2 | 3e-2 |


## Training FLOPs Measurement

`profiling/profile_flops.py` prints one table row per configuration.

```bash
cd profiling

# Inference (forward) FLOPs, conventional sizes
python3 profile_flops.py --flops-mode forward --vit-sizes tiny small base large --num-classes 100

# Measured training (forward + loss + backward) FLOPs
python3 profile_flops.py --flops-mode train-measured --vit-sizes tiny --num-classes 100

# Hourglass sweep: mlp_ratio=0.5, d_z=288, L ∈ {9..13}
python3 profile_flops.py \
  --flops-mode train-measured \
  --vit-sizes tiny --mlp-ratios 0.5 --embed-dims 288 --depths 9 10 11 12 13 --num-heads 3 \
  --num-classes 100 --device cuda:0
```

Notes:

- `train-measured` measures the true forward+backward FLOPs (close to, but not exactly,  3× the forward FLOPs).
- `--num-classes None` will profile the backbone without classification head.

To find the `d_z` that matches the baseline's training FLOPs at a new `mlp_ratio`:

```bash
python3 search_config_by_train_flops.py \
  --target-size tiny --mlp-ratio 0.5 --depths 12 11 \
  --num-classes 196 --flops-mode train-measured --device cuda:0
```

## Inference Latency Calculation

```bash
cd profiling

# ms/image and img/s, fp16, batch 32
python3 inference_latency.py \
  --vit-sizes tiny --mlp-ratios 0.5 --embed-dims 288 --depths 11 12 --num-heads 3 \
  --num-classes 100 --batch-size 32 --dtype fp16 --device cuda:0

# training-step latency (forward + backward + optimizer step)
python3 inference_latency.py --mode train-step --vit-sizes tiny --batch-size 32 --device cuda:0
```

Latency depends on GPU, dtype, and batch size; the table reports them alongside ms/image.


## Acknowledgements

- The overall training pipeline was built based on [OFDB](https://github.com/ryoo-nakamura/OFDB).
- ViT models will be loaded from the official [timm](https://github.com/huggingface/pytorch-image-models) library.
- FLOPs counting via [torch_flops](https://github.com/zugexiaodui/torch_flops) and `torch.utils.flop_counter`.
