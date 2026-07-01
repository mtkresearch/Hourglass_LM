#!/bin/bash

latent_dim=3072
hidden_dims="3546 3546 3546 3546 3546 3546"
device="cuda:8"
epochs=2
batch_size=512
lr=3e-4
ds_name="imagenet32"
mode="denoising"
model_type="conventional"


# Loop over run_id
for run_id in 1 2
do
    for lr in $lrs
    do

    echo "=========================================="
    echo "Starting experiments with run_id=${run_id}"
    echo "=========================================="


    for lr in $lrs
    do
        echo "=========================================="
        echo "Starting experiments with run_id=${lr}"
        echo "=========================================="

        # Conventional w/ W_in
        echo "Running: Conventional w/ W_in"
        python3 run.py --ds_name ${ds_name} \
                    --mode ${mode} \
                    --model_type ${model_type} \
                    --latent_dim ${latent_dim} \
                    --hidden_dims ${hidden_dims} \
                    --device ${device} \
                    --epochs ${epochs} \
                    --batch_size ${batch_size} \
                    --lr ${lr} \
                    --run_id ${run_id} \
                    --noise_std 0.25 \
                    --use_augmentation \
                    --aug_num 4 &

        # Conventional w/ W_in, W_in = I, fix 
        echo "Running: Conventional w/ W_in, W_in = I, fix"
        python3 run.py --ds_name ${ds_name} \
                    --mode ${mode} \
                    --model_type ${model_type} \
                    --latent_dim ${latent_dim} \
                    --hidden_dims ${hidden_dims} \
                    --device ${device} \
                    --epochs ${epochs} \
                    --batch_size ${batch_size} \
                    --lr ${lr} \
                    --run_id ${run_id} \
                    --fix_Win \
                    --I_Win \
                    --noise_std 0.25 \
                    --use_augmentation \
                    --aug_num 4 &

        # Conventional w/ W_in, W_in = I, trainable
        echo "Running: Conventional w/ W_in, W_in = I, trainable"
        python3 run.py --ds_name ${ds_name} \
                    --mode ${mode} \
                    --model_type ${model_type} \
                    --latent_dim ${latent_dim} \
                    --hidden_dims ${hidden_dims} \
                    --device ${device} \
                    --epochs ${epochs} \
                    --batch_size ${batch_size} \
                    --lr ${lr} \
                    --run_id ${run_id} \
                    --I_Win \
                    --noise_std 0.25 \
                    --use_augmentation \
                    --aug_num 4 &

        # Conventional w/ W_in, W_in = random, fix
        echo "Running: Conventional w/ W_in, W_in = random, fix"
        python3 run.py --ds_name ${ds_name} \
                    --mode ${mode} \
                    --model_type ${model_type} \
                    --latent_dim ${latent_dim} \
                    --hidden_dims ${hidden_dims} \
                    --device ${device} \
                    --epochs ${epochs} \
                    --batch_size ${batch_size} \
                    --lr ${lr} \
                    --run_id ${run_id} \
                    --fix_Win \
                    --noise_std 0.25 \
                    --use_augmentation \
                    --aug_num 4 &

        # Conventional w/o W_in
        echo "Running: Conventional w/o W_in"
        python3 run.py --ds_name ${ds_name} \
                    --mode ${mode} \
                    --model_type ${model_type} \
                    --latent_dim ${latent_dim} \
                    --hidden_dims ${hidden_dims} \
                    --device ${device} \
                    --epochs ${epochs} \
                    --batch_size ${batch_size} \
                    --lr ${lr} \
                    --run_id ${run_id} \
                    --wo_Win \
                    --noise_std 0.25 \
                    --use_augmentation \
                    --aug_num 4 &

        echo "=========================================="
        echo "Completed all exp for lr=${lr}"    
        echo "=========================================="
    
    done
    echo "Completed run_id=${run_id}"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="