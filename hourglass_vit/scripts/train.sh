#!/bin/bash
# =============================================================================
# Unified training script for (Hourglass) ViT.
#
# Conventional ViT:
#   DATASET=cars CUDA_VISIBLE_DEVICES=0,1 ./scripts/train.sh
#
# Hourglass ViT (custom mlp_ratio / embed_dim / depth / num_heads):
#   DATASET=cars MLP_RATIO=0.5 EMBED_DIM=288 DEPTH=11 \
#     CUDA_VISIBLE_DEVICES=0,1 ./scripts/train.sh
#
# Options (env vars):
#   DATASET      cars | cifar10 | cifar100 | flowers | voc12 | imagenet100 | places30  (required)
#   MODEL        tiny (default) | small | base | large
#   MLP_RATIO    MLP hidden ratio r = d_h / d_z          (default: 4.0)
#   EMBED_DIM    token dimension d_z                     (default: per MODEL)
#   DEPTH        number of transformer layers L          (default: per MODEL)
#   NUM_HEADS    number of attention heads               (default: per MODEL)
#   LR           base learning rate, linearly scaled by total_bs/512 inside main.py
#                (default: 3e-2, the value used in the paper)
#   TOTAL_BS     total batch size across all GPUs        (default: 768; 384 for large)
#   EPOCHS       training epochs                         (default: 1000)
#   WARMUP_EPOCHS warmup epochs                          (default: 10)
#   WEIGHT_DECAY weight decay                            (default: 1e-4)
#   DATA_ROOT    dataset parent directory                (default: <repo>/data)
#   NUM_WORKERS  dataloader workers per process          (default: 4; 20 for imagenet100/places30)
#   NGPUS        number of GPUs (default: inferred from CUDA_VISIBLE_DEVICES, else 1)
#   MASTER_PORT  rendezvous port (default: random free port)
#   WANDB_PROJECT  wandb project name (default: hourglass-vit)
#                  Set WANDB_MODE=offline (or disabled) to run without a wandb account.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HYDRA_FULL_ERROR=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

# ---------------- Model registry ----------------
MODEL=${MODEL:-tiny}
case "$MODEL" in
  tiny)  MODEL_NAME=vit_tiny_patch16_224;  DEF_DIM=192;  DEF_DEPTH=12; DEF_HEADS=3;  DEF_BS=768 ;;
  small) MODEL_NAME=vit_small_patch16_224; DEF_DIM=384;  DEF_DEPTH=12; DEF_HEADS=6;  DEF_BS=768 ;;
  base)  MODEL_NAME=vit_base_patch16_224;  DEF_DIM=768;  DEF_DEPTH=12; DEF_HEADS=12; DEF_BS=768 ;;
  large) MODEL_NAME=vit_large_patch16_224; DEF_DIM=1024; DEF_DEPTH=24; DEF_HEADS=16; DEF_BS=384 ;;
  *) echo "Unknown MODEL '$MODEL' (use tiny|small|base|large)"; exit 1 ;;
esac

MLP_RATIO=${MLP_RATIO:-4.0}
EMBED_DIM=${EMBED_DIM:-$DEF_DIM}
DEPTH=${DEPTH:-$DEF_DEPTH}
NUM_HEADS=${NUM_HEADS:-$DEF_HEADS}

# ---------------- Dataset registry ----------------
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}
DATASET=${DATASET:?"Set DATASET to one of: cars cifar10 cifar100 flowers voc12 imagenet100 places30"}
DEF_WORKERS=4
case "$DATASET" in
  cars)
    DS_NAME=CARS;        NUM_CLASSES=196; TRAIN_DIR=car_data/train;     VAL_DIR=car_data/test
    TRAIN_IMGS=8144;     VAL_IMGS=8041 ;;
  cifar10)
    DS_NAME=CIFAR10;     NUM_CLASSES=10;  TRAIN_DIR=CIFAR-10/train;     VAL_DIR=CIFAR-10/test
    TRAIN_IMGS=50000;    VAL_IMGS=10000 ;;
  cifar100)
    DS_NAME=CIFAR100;    NUM_CLASSES=100; TRAIN_DIR=CIFAR-100/train;    VAL_DIR=CIFAR-100/test
    TRAIN_IMGS=50000;    VAL_IMGS=10000 ;;
  flowers)
    # Oxford Flowers-102: the large official "test" split (6149 imgs) is used
    # for training and the official "val" split (1020 imgs) for evaluation.
    DS_NAME=Flowers;     NUM_CLASSES=102; TRAIN_DIR=Flowers/test;       VAL_DIR=Flowers/val
    TRAIN_IMGS=6149;     VAL_IMGS=1020 ;;
  voc12)
    DS_NAME=VOC12;       NUM_CLASSES=20;  TRAIN_DIR=VOC-12/train;       VAL_DIR=VOC-12/val
    TRAIN_IMGS=13609;    VAL_IMGS=13841 ;;
  imagenet100)
    DS_NAME=ImageNet100; NUM_CLASSES=100; TRAIN_DIR=ImageNet-100/train; VAL_DIR=ImageNet-100/val
    TRAIN_IMGS=117000;   VAL_IMGS=13000;  DEF_WORKERS=20 ;;
  places30)
    DS_NAME=Places30;    NUM_CLASSES=30;  TRAIN_DIR=Places-30/train;    VAL_DIR=Places-30/val
    TRAIN_IMGS=149254;   VAL_IMGS=3000;   DEF_WORKERS=20 ;;
  *) echo "Unknown DATASET '$DATASET'"; exit 1 ;;
esac
NUM_WORKERS=${NUM_WORKERS:-$DEF_WORKERS}

TRAIN_ROOT=$DATA_ROOT/$TRAIN_DIR
VAL_ROOT=$DATA_ROOT/$VAL_DIR
for d in "$TRAIN_ROOT" "$VAL_ROOT"; do
  if [ ! -d "$d" ]; then
    echo "❌ Dataset directory not found: $d"
    echo "   Prepare the dataset first (see data_prep/README.md) or set DATA_ROOT."
    exit 1
  fi
done
# hydra changes the working directory to output_dir at runtime, so make paths absolute
TRAIN_ROOT=$(readlink -f "$TRAIN_ROOT")
VAL_ROOT=$(readlink -f "$VAL_ROOT")

# ---------------- Training hyperparameters ----------------
LR=${LR:-3e-2}
TOTAL_BS=${TOTAL_BS:-$DEF_BS}
EPOCHS=${EPOCHS:-1000}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}
WEIGHT_DECAY=${WEIGHT_DECAY:-1.0e-4}
export WANDB_PROJECT=${WANDB_PROJECT:-hourglass-vit}

# ---------------- GPU count & per-GPU batch size ----------------
if [ -z "${NGPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
  else
    NGPUS=1
  fi
fi
if [ $((TOTAL_BS % NGPUS)) -ne 0 ]; then
  echo "❌ TOTAL_BS ($TOTAL_BS) is not divisible by NGPUS ($NGPUS)."
  exit 1
fi
PER_BS=$((TOTAL_BS / NGPUS))

# ---------------- Experiment name / output dir ----------------
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
EXP_NAME="${DS_NAME}_${MODEL_NAME}_mlp${MLP_RATIO}_dz${EMBED_DIM}_L${DEPTH}_h${NUM_HEADS}_lr${LR}_bs${TOTAL_BS}_${TIMESTAMP}"
OUTPUT_DIR=./output/${EXP_NAME}

echo "==================== Config ===================="
echo "Dataset:       $DS_NAME ($NUM_CLASSES classes)"
echo "  train root:  $TRAIN_ROOT ($TRAIN_IMGS imgs)"
echo "  val root:    $VAL_ROOT ($VAL_IMGS imgs)"
echo "Model:         $MODEL_NAME"
echo "  mlp_ratio:   $MLP_RATIO"
echo "  embed_dim:   $EMBED_DIM"
echo "  depth:       $DEPTH"
echo "  num_heads:   $NUM_HEADS"
echo "LR (base):     $LR  (scaled by ${TOTAL_BS}/512 inside main.py)"
echo "Batch size:    $TOTAL_BS total / $NGPUS GPUs = $PER_BS per GPU"
echo "Epochs:        $EPOCHS (warmup $WARMUP_EPOCHS)"
echo "Output dir:    $OUTPUT_DIR"
echo "================================================="

# Port 0 lets torchrun pick a random free port, so concurrent runs never clash.
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:${MASTER_PORT:-0} \
  --nproc-per-node=$NGPUS \
  main.py data=colorimagefolder \
  data.baseinfo.name=$DS_NAME \
  data.baseinfo.num_classes=$NUM_CLASSES \
  data.trainset.root=$TRAIN_ROOT \
  data.baseinfo.train_imgs=$TRAIN_IMGS \
  data.valset.root=$VAL_ROOT \
  data.baseinfo.val_imgs=$VAL_IMGS \
  data.loader.batch_size=$PER_BS \
  data.loader.num_workers=$NUM_WORKERS \
  model=vit \
  model.arch.model_name=$MODEL_NAME \
  +model.arch.mlp_ratio=$MLP_RATIO \
  +model.arch.depth=$DEPTH \
  +model.arch.embed_dim=$EMBED_DIM \
  +model.arch.num_heads=$NUM_HEADS \
  model.optim.opt=sgd \
  model.optim.lr=$LR \
  model.optim.weight_decay=$WEIGHT_DECAY \
  model.scheduler.args.warmup_epochs=$WARMUP_EPOCHS \
  epochs=$EPOCHS \
  logger.project=$WANDB_PROJECT \
  logger.exp_name=$EXP_NAME \
  output_dir=$OUTPUT_DIR
