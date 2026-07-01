Run $d_{z} \in \{3075, 3100, 3200, 3300, 3400, 3546\}$, $d_{h}=270$, $L=5$
```bash
# Hourglass
PYTHONUNBUFFERED=1 ./rebuttal/linear_separability/run.sh 2>&1 | tee ./rebuttal/linear_separability/run.txt

PYTHONUNBUFFERED=1 ./rebuttal/linear_separability/run_2.sh 2>&1 | tee ./rebuttal/linear_separability/run_2.txt

PYTHONUNBUFFERED=1 ./rebuttal/linear_separability/run_ep8.sh 2>&1 | tee ./rebuttal/linear_separability/run_ep8.txt


# Hourglass (fixed W_in)
PYTHONUNBUFFERED=1 ./rebuttal/linear_separability/run_fix.sh 2>&1 | tee ./rebuttal/linear_separability/run_fix.txt

PYTHONUNBUFFERED=1 ./rebuttal/linear_separability/run_fix_2.sh 2>&1 | tee ./rebuttal/linear_separability/run_fix_2.txt

PYTHONUNBUFFERED=1 ./rebuttal/linear_separability/run_fix_3.sh 2>&1 | tee ./rebuttal/linear_separability/run_fix_3.txt

PYTHONUNBUFFERED=1 ./rebuttal/linear_separability/run_fix_4.sh 2>&1 | tee ./rebuttal/linear_separability/run_fix_4.txt

PYTHONUNBUFFERED=1 ./rebuttal/linear_separability/run_fix_ep8.sh 2>&1 | tee ./rebuttal/linear_separability/run_fix_ep8.txt
```


Plot
```bash
# 3075
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_27195300/latent3075_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_17748900/latent3075_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3075.png 

# 3100
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_27416400/latent3100_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_17893200/latent3100_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3100.png 

# 3200
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_28300800/latent3200_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_18470400/latent3200_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3200.png 

# 3300
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_29185200/latent3300_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_19047600/latent3300_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3300.png 

# 3400
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_30069600/latent3400_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_19624800/latent3400_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3400.png 

# 3546
## epoch2 
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_31360824/latent3546_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_20467512/latent3546_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3546.png 
## epoch4
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep4_aug4/hourglass_w_Win_31360824/latent3546_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep4_aug4/hourglass_w_Win_fix_20467512/latent3546_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3546_ep4.png 

# 3700
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_32722800/latent3700_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_21356400/latent3700_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3700.png 

# 3800
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_33607200/latent3800_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_21933600/latent3800_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_3800.png 

# 4000
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_35376000/latent4000_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_23088000/latent4000_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_4000.png 

# 4500
python3 ./rebuttal/linear_separability/plot.py \
--hourglass_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_39798000/latent4500_hidden270_270_270_270_270 \
--hourglass_fz_path ./results/imagenet32_denoising_std0.25/bs512_ep2_aug4/hourglass_w_Win_fix_25974000/latent4500_hidden270_270_270_270_270 \
--save_path ./rebuttal/linear_separability/comparison_imagenet32_denoising_4500.png 

``` 