#!/bin/bash

latent_dim=3072
epochs=2
batch_size=512
lrs=(1e-4 3e-4 5e-3 7e-4)
device="cuda:1"


hidden_dim_values=(3075 3546 4012 4576)

for run_id in {1..2}; do
    for repeat in {1..5}; do
        
        for dim_value in "${hidden_dim_values[@]}"; do
                    
            hidden_dims=""
            for ((i=1; i<=repeat; i++)); do
                hidden_dims="$hidden_dims $dim_value"
            done
            hidden_dims=$(echo $hidden_dims | xargs)
            
            echo "Running: run_id=$run_id, hidden_dims=[$hidden_dims], layers=$repeat"
            
            for lr in "${lrs[@]}"; do
                
                python3 -u ./run_new.py \
                    --ds_name imagenet32 \
                    --mode denoising \
                    --model_type conventional \
                    --latent_dim "$latent_dim" \
                    --epochs "$epochs" \
                    --batch_size "$batch_size" \
                    --hidden_dims $hidden_dims \
                    --lr "$lr" \
                    --device "$device" \
                    --run_id "$run_id" \
                    --use_augmentation \
                    --aug_num 4 \
                    --wo_Win \
                    --wo_Wout &
                
            done
            wait
            
        done
    done
done

echo "All experiments completed!"