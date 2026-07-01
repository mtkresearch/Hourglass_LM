#!/bin/bash

latent_dim=784
# epochs=50
epochs=100
batch_size=128
lrs=(1e-4 5e-4 1e-3 5e-3)
device="cuda:2"
aug_num=4

hidden_dim_values=(785 905 1150 1470 2870)

for run_id in {1..5}; do
    
    for dim_value in "${hidden_dim_values[@]}"; do
        
        for repeat in {1..6}; do
            
            hidden_dims=""
            for ((i=1; i<=repeat; i++)); do
                hidden_dims="$hidden_dims $dim_value"
            done
            hidden_dims=$(echo $hidden_dims | xargs)
            
            echo "Running: run_id=$run_id, hidden_dims=[$hidden_dims], layers=$repeat"
            
            for lr in "${lrs[@]}"; do
                
                python3 ./run.py \
                    --ds_name mnist \
                    --mode generative_classification \
                    --model_type conventional \
                    --latent_dim "$latent_dim" \
                    --epochs "$epochs" \
                    --batch_size "$batch_size" \
                    --hidden_dims $hidden_dims \
                    --lr "$lr" \
                    --device "$device" \
                    --run_id "$run_id" \
                    --use_augmentation \
                    --aug_num "$aug_num" &
                
            done
            wait
            
        done
    done
done

echo "All experiments completed!"