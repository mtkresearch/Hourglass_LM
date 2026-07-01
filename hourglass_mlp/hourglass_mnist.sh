#!/bin/bash
set -euo pipefail

# 基本配置參數
ds_name="mnist"
mode="super_resolution"
model_type="hourglass"
epochs=100
batch_size=128
aug_num=4
device="cuda:0"
MAX_PARALLEL=10
RUNS_PER_CFG=3

# 實驗參數範圍
latent_dims=(785 1150 1470 1790 2140)
hidden_dim_values=(16 32 64 115 270 350 480 860 1048)
lrs=(1e-4 5e-4 1e-3 5e-3)
min_layer=2
max_layer=10
layer_interval=2

min_layer=10
max_layer=50
layer_interval=10


trap 'echo "捕獲中斷信號，清理進程..."; kill $(jobs -p) 2>/dev/null || true' INT TERM EXIT

run_experiment() {
    local latent_dim=$1   
    local hidden_dim=$2
    local layers=$3
    local lr=$4
    local run_id=$5

    # 創建hidden_dims字串，以空格分隔
    local hidden_dims=""
    for ((i=1; i<=layers; i++)); do
        hidden_dims+="$hidden_dim "
    done
    hidden_dims=${hidden_dims% }

    echo "[RUN] latent_dim=${latent_dim}, hidden_dims=${hidden_dims}, layers=${layers}, lr=${lr}, run_id=${run_id}"

    python3 -u ./run.py \
        --ds_name "$ds_name" \
        --mode "$mode" \
        --model_type "$model_type" \
        --latent_dim "$latent_dim" \
        --epochs $epochs \
        --batch_size "$batch_size" \
        --use_augmentation \
        --aug_num "$aug_num" \
        --hidden_dims $hidden_dims \
        --lr "$lr" \
        --device "$device" \
        --run_id "$run_id"
}

# 主實驗循環及並行管理
running=0
total_jobs=0

# 外部循環 - 每個配置重複執行的次數
for run_id in $(seq 1 $RUNS_PER_CFG); do
    echo "準備開始實驗 Run ID: $run_id"

    for latent_dim in "${latent_dims[@]}"; do
        for hidden_dim in "${hidden_dim_values[@]}"; do
            for layers in $(seq $min_layer $layer_interval $max_layer); do
                for lr in "${lrs[@]}"; do
                    # 如果達到最大並行作業數，等待
                    while (( running >= MAX_PARALLEL )); do
                        wait -n 2>/dev/null || true
                        running=$((running-1))
                    done
                    
                    # 在背景啟動作業
                    run_experiment "$latent_dim" "$hidden_dim" "$layers" "$lr" "$run_id" &
                    running=$((running+1))
                    total_jobs=$((total_jobs+1))
                    
                    echo "[LAUNCH] Job $total_jobs (running=$running)"
                    sleep 0.4  # 防止系統負載過重
                done
            done
        done
    done
done

# 等待剩餘作業完成
while (( running > 0 )); do
    wait -n 2>/dev/null || true
    running=$((running-1))
    echo "[FINISH] Jobs remaining: $running"
done

echo "所有實驗完成！總作業數: $total_jobs"