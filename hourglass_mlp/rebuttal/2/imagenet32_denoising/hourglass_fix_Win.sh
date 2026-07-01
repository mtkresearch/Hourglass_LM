#!/bin/bash

epochs=2
batch_size=512
device="cuda:0"

# Define the configurations
configurations=(
    # "3546 8 5"
    # "3546 16 5"
    # "3546 64 5"
    # "3546 128 5"
    # "3546 270 3"
    # "3075 765 2"
    # "3546 270 4"
    # "3546 270 5"
    "3075 765 3"
    "3546 270 7"
    "3546 270 8"
    "3075 765 4"
    "3075 765 5"
    "3075 1146 4"
    "3075 1146 5"
    "3075 1560 4"
    "3546 1146 5"
    "3546 1560 4"
    "3075 1560 5"
    "3546 1560 5"
    "3075 2014 5"
)

for run_id in $(seq 1 2); do
    for config in "${configurations[@]}"; do
        IFS=' ' read -r -a params <<< "$config"
        latent_dim="${params[0]}"
        hidden_dim="${params[1]}"
        L="${params[2]}"

        # hidden_dims=$(printf "%${L}d" 0 | tr '0' ' ')$hidden_dim
        hidden_dims=$(yes "$hidden_dim" | head -n "$L" | paste -sd " " -)
        echo "Running: run_id=$run_id, hidden_dims=[$hidden_dims], latent_dim=$latent_dim, L=$L"

        for lr in 1e-4 3e-4 5e-4 7e-4 1e-3; do
            python3 -u ./run_new.py \
                --ds_name imagenet32 \
                --mode denoising \
                --model_type hourglass \
                --latent_dim "$latent_dim" \
                --epochs "$epochs" \
                --batch_size "$batch_size" \
                --hidden_dims $hidden_dims \
                --lr "$lr" \
                --device "$device" \
                --run_id "$run_id" \
                --use_augmentation \
                --aug_num 4 \
                --fix_Win &
        done
        wait
    done
done

echo "All experiments completed!"
