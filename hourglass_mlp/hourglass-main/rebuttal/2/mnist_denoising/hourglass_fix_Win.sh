#!/bin/bash

epochs=30
batch_size=128
device="cuda:8"
parallel_jobs=6
ds_name="mnist"
model_type="hourglass"
mode="denoising"

# run_ids 範圍
# run_ids=(1 2 3 4 5)
run_ids=(1)

# learning rates
lrs=(0.0001 0.0005 0.001 0.005 0.01)

# 定義 Frontier 配置 (格式: "latent_dim hidden_dim L")，移除 lr
configs=(
    # "785 4 5"
    # "785 4 10"
    # "785 4 15"
    # "785 4 20"
    # "785 4 25"
    # "785 115 1"
    # "785 4 35"
    # "785 4 45"
    # "785 115 2"
    # "785 115 3"
    # "785 115 4"
    # "785 115 5"
    # "785 115 6"
    # "785 115 7"
    # "785 115 9"
    # "785 115 10"
    # "785 270 5"
    # "785 270 6"
    # "785 270 7"
    # "1470 115 6"
    # "1470 115 7"
    # "1470 115 8"
    # "1470 115 9"
    # "2120 115 8"
    # "4096 64 8"
    # "2120 480 4"
    "4096 240 4"
)

# 用於追蹤所有子進程
pids=()

# Ctrl+C 信號處理函數
cleanup() {
    echo ""
    echo "=========================================="
    echo "Caught Ctrl+C! Terminating all processes..."
    echo "=========================================="
    
    # 終止所有子進程
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    
    # 等待 5 秒讓進程正常終止
    sleep 5
    
    # 強制終止還活著的進程
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    
    echo "All processes terminated."
    exit 1
}

# 註冊信號處理
trap cleanup SIGINT SIGTERM

count=0

for run_id in "${run_ids[@]}"; do
    echo "=========================================="
    echo "Starting run_id: $run_id"
    echo "=========================================="
    
    for config in "${configs[@]}"; do
        read -r latent_dim hidden_dim L <<< "$config"
        
        # 構建 hidden_dims (重複 L 次)
        hidden_dims=$(printf "%s " $(yes "$hidden_dim" | head -n "$L"))
        
        # 對每個 lr 執行
        for lr in "${lrs[@]}"; do
            echo "Running: run_id=$run_id, latent=$latent_dim, hidden=$hidden_dim, L=$L, lr=$lr"
            
            python3 ./run_new.py \
                --ds_name "$ds_name" \
                --mode "$mode" \
                --model_type "$model_type" \
                --latent_dim "$latent_dim" \
                --epochs "$epochs" \
                --batch_size "$batch_size" \
                --hidden_dims $hidden_dims \
                --lr "$lr" \
                --device "$device" \
                --run_id "$run_id" \
                --fix_Win &
            
            # 記錄子進程 PID
            pids+=($!)
            
            ((count++))
            
            # 每 parallel_jobs 個任務等待一次
            if [ $((count % parallel_jobs)) -eq 0 ]; then
                wait
                # 清理已完成的 PIDs
                pids=()
            fi
        done
    done
done

# 等待剩餘任務完成
wait

echo "=========================================="
echo "All experiments completed!"
echo "Total runs: ${#run_ids[@]} x ${#configs[@]} x ${#lrs[@]} = $((${#run_ids[@]} * ${#configs[@]} * ${#lrs[@]}))"
echo "=========================================="