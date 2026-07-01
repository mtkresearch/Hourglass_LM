#!/bin/bash
# latent_dims="3075 3100 3200 3300 3400 3546"
# latent_dims="3546 3700 3800 4000 4500"
latent_dim="3546"
# lrs="5e-4"
lrs="3e-4"
device='cuda:4'
# run_id=5
epochs=2
# epochs=4

for lr in $lrs; do
    # for run_id in {1..5}; do
    # for run_id in {11..15}; do
    for run_id in {16..16}; do
    # for latent_dim in $latent_dims; do
        echo "Running with learning rate: $lr"

        python3 ./run.py \
            --ds_name 'imagenet32' \
            --mode 'denoising' \
            --noise_std 0.25 \
            --model_type 'hourglass' \
            --latent_dim "$latent_dim"   \
            --epochs "$epochs" \
            --batch_size 512 \
            --use_augmentation \
            --aug_num 4 \
            --hidden_dims 270 270 270 270 270 \
            --lr "$lr" \
            --run_id "$run_id" \
            --device "$device" &
    done
    wait
done

echo "All experiments completed!"