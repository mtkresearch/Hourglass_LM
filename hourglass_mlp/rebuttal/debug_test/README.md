

```bash
cd ./rebuttal/debug_test
```

## Ablation (old)
```bash
python3 -u ./run_ablation.py \
    --ds_name imagenet32 \
    --mode denoising \
    --model_type hourglass \
    --latent_dim 3546 \
    --epochs 2 \
    --batch_size 512 \
    --hidden_dims 270 270 270 270 270 \
    --lr 5e-4 \
    --device "cuda:4" \
    --run_id 21 \
    --use_augmentation \
    --aug_num 4 \
    --freeze_hg_in_out 2>&1 | tee ./rebuttal/debug_test/ablation_old.txt
```


## Ablation (new)
```bash
python3 -u ./run.py \
    --ds_name imagenet32 \
    --mode denoising \
    --model_type hourglass \
    --latent_dim 3546 \
    --epochs 2 \
    --batch_size 512 \
    --hidden_dims 270 270 270 270 270 \
    --lr 3e-4 \
    --device "cuda:4" \
    --run_id 21 \
    --use_augmentation \
    --aug_num 4 2>&1 | tee ./rebuttal/debug_test/new.txt
```
