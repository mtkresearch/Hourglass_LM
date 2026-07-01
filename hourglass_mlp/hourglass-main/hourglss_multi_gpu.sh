#!/bin/bash
set -euo pipefail

# 基本配置參數
ds_name="mnist"
mode="super_resolution"ㄕ
model_type="hourglass"
epochs=100
batch_size=128
aug_num=4
MAX_PARALLEL_PER_GPU=10  # 每個GPU的最大並行進程數
GPU_COUNT=4              # 可用GPU數量，從cuda:0到cuda:(GPU_COUNT-1)
RUNS_PER_CFG=3

# 實驗參數範圍
latent_dims=(785 1150 1470 1790 2140)
hidden_dim_values=(16 32 64 115 270 350 480 860 1048)
lrs=(1e-4 5e-4 1e-3 5e-3)
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
    local gpu_id=$6

    # 創建hidden_dims字串，以空格分隔
    local hidden_dims=""
    for ((i=1; i<=layers; i++)); do
        hidden_dims+="$hidden_dim "
    done
    hidden_dims=${hidden_dims% }

    local device="cuda:$gpu_id"
    echo "[RUN] latent_dim=${latent_dim}, hidden_dims=${hidden_dims}, layers=${layers}, lr=${lr}, run_id=${run_id}, device=${device}"

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

# 創建用於跟踪每個GPU上運行的作業數的陣列
declare -a running_per_gpu
for ((i=0; i<GPU_COUNT; i++)); do
    running_per_gpu[$i]=0
done

total_running=0
total_jobs=0

# 找到最不忙的GPU
get_least_busy_gpu() {
    local min_jobs=${running_per_gpu[0]}
    local min_gpu=0
    
    for ((i=1; i<GPU_COUNT; i++)); do
        if ((${running_per_gpu[$i]} < min_jobs)); then
            min_jobs=${running_per_gpu[$i]}
            min_gpu=$i
        fi
    done
    
    echo $min_gpu
}

# 等待至少一個GPU有空閒槽位
wait_for_gpu_slot() {
    while ((total_running >= GPU_COUNT * MAX_PARALLEL_PER_GPU)); do
        wait -n 2>/dev/null || true
        total_running=$((total_running-1))
        
        # 找到剛剛完成的作業是哪個GPU的
        for ((i=0; i<GPU_COUNT; i++)); do
            if ((${running_per_gpu[$i]} > 0)); then
                running_per_gpu[$i]=$((${running_per_gpu[$i]}-1))
                break
            fi
        done
    done
}

# 主實驗循環及並行管理
for run_id in $(seq 1 $RUNS_PER_CFG); do
    echo "準備開始實驗 Run ID: $run_id"

    for latent_dim in "${latent_dims[@]}"; do
        for hidden_dim in "${hidden_dim_values[@]}"; do
            for layers in $(seq $min_layer $layer_interval $max_layer); do
                for lr in "${lrs[@]}"; do
                    # 等待有空閒GPU槽位
                    wait_for_gpu_slot
                    
                    # 選擇最不忙的GPU
                    gpu_id=$(get_least_busy_gpu)
                    
                    # 在背景啟動作業
                    run_experiment "$latent_dim" "$hidden_dim" "$layers" "$lr" "$run_id" "$gpu_id" &
                    
                    # 更新計數器
                    running_per_gpu[$gpu_id]=$((${running_per_gpu[$gpu_id]}+1))
                    total_running=$((total_running+1))
                    total_jobs=$((total_jobs+1))
                    
                    echo "[LAUNCH] Job $total_jobs on GPU $gpu_id (GPU $gpu_id: ${running_per_gpu[$gpu_id]} jobs, Total: $total_running jobs)"
                    sleep 0.4  # 防止系統負載過重
                done
            done
        done
    done
done

# 等待所有剩餘作業完成
echo "等待所有作業完成..."
wait
echo "所有實驗完成！總作業數: $total_jobs"