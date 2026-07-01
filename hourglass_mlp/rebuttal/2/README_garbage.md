## Below is draft


[TODO]
+ 確認paper Figure 3.a 中的點要怎麼reproduce `Done`
+ Copy 舊的 raw results 到這邊，並且轉換格式
+ 跑 conventional w/o W_in, W_out
+ 把畫圖函數整理好放到這個folder底下


## 確認paper Figure 3.a 中的點要怎麼reproduce


```bash
# conventional w/ W_in
python3 ./run.py \
    --ds_name mnist \
    --mode denoising \
    --model_type conventional \
    --latent_dim 784 \
    --epochs 30 \
    --batch_size 128 \
    --hidden_dims 785 785 \
    --lr 1e-3 \
    --device "cuda:3" \
    --run_id 1

# 目標: 24.25225089416504 --> 紀錄到的是 final eval PSNR (有喔這個我們都有記錄到!)
# 重跑: 
    # final eval: [24.2663, 24.2412, 24.2463]
    # test : [24.2840, 24.2880, 24.2920]

# conventional w/o W_in w/o W_out
python3 ./run_new.py \
    --ds_name mnist \
    --mode denoising \
    --model_type conventional \
    --latent_dim 784 \
    --epochs 30 \
    --batch_size 128 \
    --hidden_dims 785 785 \
    --lr 1e-3 \
    --device "cuda:3" \
    --wo_Win \
    --wo_Wout \
    --run_id 1
```


## 跑 conventional w/o W_in, W_out Grid Search 並畫圖
### MNIST Generative Classification
run
```bash
chmod +x ./rebuttal/2/mnist_generative_classification/conventional.sh
PYTHONUNBUFFERED=1 ./rebuttal/2/mnist_generative_classification/conventional.sh 2>&1 | tee ./rebuttal/2/mnist_generative_classification/conventional.txt

chmod +x ./rebuttal/2/mnist_generative_classification/conventional_wo_in_out.sh
PYTHONUNBUFFERED=1 ./rebuttal/2/mnist_generative_classification/conventional_wo_in_out.sh 2>&1 | tee ./rebuttal/2/mnist_generative_classification/conventional_wo_in_out.txt
```

get results
```bash
# python3 ./rebuttal/2/mnist_generative_classification/transform_json.py ./old/src/exp_record/mnist/generative_classification/mnist_generative_classification.json ./rebuttal/2/mnist_generative_classification/old_results.json

python3 ./rebuttal/2/get_results.py --exp_loc 'results' --exp_folder 'mnist_generative_classification' --json_output './rebuttal/2/mnist_generative_classification'

python3 ./rebuttal/2/combine_json.py ./rebuttal/2/mnist_generative_classification/all.json ./rebuttal/2/mnist_generative_classification/old_results.json ./rebuttal/2/mnist_generative_classification/mnist_generative_classification.json
```

plot
```bash
python3 ./rebuttal/2/plot_frontier.py \
 --json_path ./rebuttal/2/mnist_generative_classification/all.json \
  --metric 'test_psnr' --plot_output ./rebuttal/2/mnist_generative_classification --x_min 1.0 --x_max 8 --title '' --bs_ep 'bs128_ep50' --show_arch

python3 ./rebuttal/2/plot_frontier.py \
 --json_path ./rebuttal/2/mnist_generative_classification/all.json \
  --metric 'test_psnr' --plot_output ./rebuttal/2/mnist_generative_classification --x_min 1.0 --x_max 8 --title '' --bs_ep 'bs128_ep100_aug4' --show_arch

```


### MNIST Denoising
run
```bash
chmod +x ./rebuttal/2/mnist_denoising/conventional_wo_in_out.sh
PYTHONUNBUFFERED=1 ./rebuttal/2/mnist_denoising/conventional_wo_in_out.sh 2>&1 | tee ./rebuttal/2/mnist_denoising/conventional_wo_in_out.txt

chmod +x ./rebuttal/2/mnist_denoising/conventional.sh
PYTHONUNBUFFERED=1 ./rebuttal/2/mnist_denoising/conventional.sh 2>&1 | tee ./rebuttal/2/mnist_denoising/conventional.txt

chmod +x ./rebuttal/2/mnist_denoising/conventional_wo_in_out.sh
PYTHONUNBUFFERED=1 ./rebuttal/2/mnist_denoising/conventional_wo_in_out.sh 2>&1 | tee ./rebuttal/2/mnist_denoising/conventional_wo_in_out.txt

```

get results
```bash
# python3 ./rebuttal/2/transform_json.py ./old/src/exp_record/mnist/denoising_std0.25_man/mnist_denoising_std0.25.json ./rebuttal/2/mnist_denoising/old_results.json

python3 ./rebuttal/2/get_results.py --exp_loc 'results' --exp_folder 'mnist_denoising_std0.25' --json_output './rebuttal/2/mnist_denoising'

python3 ./rebuttal/2/combine_json.py ./rebuttal/2/mnist_denoising/all.json ./rebuttal/2/mnist_denoising/old_results.json ./rebuttal/2/mnist_denoising/mnist_denoising_std0.25.json
```

plot
```bash
python3 ./rebuttal/2/plot_frontier.py \
 --json_path ./rebuttal/2/mnist_denoising/all.json \
  --metric 'eval_psnr' --plot_output ./rebuttal/2/mnist_denoising --x_min 1.85 --x_max 8 --title '' --bs_ep 'bs128_ep30' --std_scale 0.2 --show_arch
```
[Note]: 這裡在paper中，std_scale用到 0.2 倍，且不小心看到eval psnr

```bash
python3 ./rebuttal/2/plot_frontier.py \
 --json_path ./rebuttal/2/mnist_denoising/all.json \
  --metric 'test_psnr' --plot_output ./rebuttal/2/mnist_denoising --x_min 1.0 --x_max 15.0 --title '' --bs_ep 'bs128_ep100_aug4' --show_arch
``` 


Old resuls in paper:
```
  Frontier (x, mean, std, group, runs):
    x=2.460M, mean=24.229, std=0.000, 5*std=0.000,  group=reps784_mid785|lr=0.001, runs=[24.229]
    x=2.648M, mean=24.240, std=0.000, 5*std=0.000,  group=reps784_mid905|lr=0.001, runs=[24.240]
    x=3.033M, mean=24.259, std=0.000, 5*std=0.000,  group=reps784_mid1150|lr=0.001, runs=[24.259]
    x=3.534M, mean=24.277, std=0.000, 5*std=0.000,  group=reps784_mid1470|lr=0.001, runs=[24.277]
    x=5.729M, mean=24.316, std=0.000, 5*std=0.000,  group=reps784_mid2870|lr=0.001, runs=[24.316]
    x=7.737M, mean=24.329, std=0.000, 5*std=0.000,  group=reps784_mid4150|lr=0.001, runs=[24.329]
```

### MNIST Denoising
run
```bash
chmod +x ./rebuttal/2/mnist_denoising/conventional_wo_in_out.sh
PYTHONUNBUFFERED=1 ./rebuttal/2/mnist_denoising/conventional_wo_in_out.sh 2>&1 | tee ./rebuttal/2/mnist_denoising/conventional_wo_in_out.txt

chmod +x ./rebuttal/2/mnist_denoising/conventional.sh
PYTHONUNBUFFERED=1 ./rebuttal/2/mnist_denoising/conventional.sh 2>&1 | tee ./rebuttal/2/mnist_denoising/conventional.txt

chmod +x ./rebuttal/2/mnist_denoising/conventional_2.sh
PYTHONUNBUFFERED=1 ./rebuttal/2/mnist_denoising/conventional_2.sh 2>&1 | tee ./rebuttal/2/mnist_denoising/conventional_2.txt
```

get results
```bash
# python3 ./rebuttal/2/transform_json.py ./old/src/exp_record/mnist/denoising_std0.25_man/mnist_denoising_std0.25.json ./rebuttal/2/mnist_denoising/old_results.json

python3 ./rebuttal/2/get_results.py --exp_loc 'results' --exp_folder 'mnist_denoising_std0.25' --json_output './rebuttal/2/mnist_denoising'

python3 ./rebuttal/2/combine_json.py ./rebuttal/2/mnist_denoising/all.json ./rebuttal/2/mnist_denoising/old_results.json ./rebuttal/2/mnist_denoising/mnist_denoising_std0.25.json
```

plot
```bash
python3 ./rebuttal/2/plot_frontier.py \
 --json_path ./rebuttal/2/mnist_denoising/all.json \
  --metric 'eval_psnr' --plot_output ./rebuttal/2/mnist_denoising --x_min 1.85 --x_max 8 --title '' --bs_ep 'bs128_ep30' --std_scale 0.2
``` 



### Imagenet32 Denoising
run
```bash
# python3 ./rebuttal/2/imagenet32_denoising/hourglass_fix_Win.sh
nohup ./rebuttal/2/imagenet32_denoising/hourglass_fix_Win.sh 2>&1 | tee ./rebuttal/2/imagenet32_denoising/hourglass_fix_Win.txt &

screen ls
screen -S myjob
screen -r myjob
./rebuttal/2/imagenet32_denoising/hourglass_fix_Win.sh 2>&1 | tee ./rebuttal/2/imagenet32_denoising/hourglass_fix_Win.txt
ctrl + A 然後 D to detach

screen -S myjob_2
screen -r myjob_2
./rebuttal/2/imagenet32_denoising/conventional_wo_Win_wo_Wout.sh 2>&1 | tee ./rebuttal/2/imagenet32_denoising/conventional_wo_Win_wo_Wout.txt

screen -S myjob_3
screen -r myjob_3
./rebuttal/2/imagenet32_denoising/hourglass_fix_Win_2.sh 2>&1 | tee ./rebuttal/2/imagenet32_denoising/hourglass_fix_Win_2.txt
```

get results
```bash
# python3 ./rebuttal/2/imagenet32_denoising/transform_json.py ./old/src/exp_record/imagenet32/bs512_ep2_denoising_std0.25_aug4/imagenet32_denoising_std0.25_aug4.json ./rebuttal/2/imagenet32_denoising/old_results.json

python3 ./rebuttal/2/get_results.py --exp_loc 'results' --exp_folder 'imagenet32_denoising_std0.25' --json_output './rebuttal/2/imagenet32_denoising'

python3 ./rebuttal/2/combine_json.py ./rebuttal/2/imagenet32_denoising/all.json ./rebuttal/2/imagenet32_denoising/old_results.json ./rebuttal/2/imagenet32_denoising/imagenet32_denoising_std0.25.json
```

plot
```bash
python3 ./rebuttal/2/plot_frontier.py --json_path ./rebuttal/2/imagenet32_denoising/all.json --metric 'test_psnr' --plot_output ./rebuttal/2/imagenet32_denoising --x_min 18 --x_max 90 --title '' --bs_ep 'bs512_ep2_aug4'
``` 



### Imagenet 32 Super resolution

Get results
```bash
python3 ./rebuttal/2/imagenet32_super_resolution/transform_json.py ./old/src/exp_record/imagenet32/bs512_ep2_super_down2.0_aug4/imagenet32_super_down2.0_aug4.json ./rebuttal/2/imageet32_super_resolution/old_results.json


```



### MNIST Super resolution

Get results
```bash
python3 ./rebuttal/2/imagenet32_super_resolution/transform_json.py ./old/src/exp_record/mnist/bs64_ep50_super_down2.0_aug4/mnist_super_down2.0_aug4.json ./rebuttal/2/mnist_super_resolution/old_results.json

```