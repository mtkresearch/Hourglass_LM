#!/bin/bash

latent_dim=784
hidden_dims="1150 1150 1150 1150 1150 1150"
device="cuda:1"
epochs=200
batch_size=128
# lrs="1e-4 5e-4 1e-3 5e-3"
# lrs="5e-4 1e-4"
lrs="5e-5 1e-4 5e-4 5e-4"
ds_name="mnist"
mode="generative_classification"
model_type="conventional"
aug_num=2

for run_id in 1 2 3 4 5 
do
    for lr in $lrs
    do
        echo "=========================================="
        echo "Starting experiments with run_id=${run_id}"
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
                    --use_augmentation \
                    --aug_num ${aug_num} &

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
                    --use_augmentation \
                    --aug_num ${aug_num} &

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
                    --use_augmentation \
                    --aug_num ${aug_num} &


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
                    --use_augmentation \
                    --aug_num ${aug_num} &

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
                    --use_augmentation \
                    --aug_num ${aug_num} &

        echo "=========================================="
        echo "Completed all exp for lr=${lr}"    
        echo "=========================================="
        wait 
    done
    echo "=========================================="
    echo "Completed run_id=${run_id}"    
    echo "=========================================="
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="