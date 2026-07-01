# Ablation: Conventional MLP — w/ $W_{in}$ vs. w/o $W_{in}$


## 1.1. [Bug] Redo Paper Fig.5

+ Bugs found when doing Fig.5:
  + 沒有使用最佳LR (`5e-4`)，而使用到 `3e-4`
  + 做Fig.5時使用的 `train_and_eval` 函數與其他主實驗地方不同
  + 在此`train_and_eval`中，`model.eval()` 狀態在某些時候，evaluation結束時沒有改回 `model.train()` 狀態 -> 造成實際trainig steps不對。

+ Original figure in the paper:

<p align="center">
  <img src="../linear_separability/comparison_imagenet_freeze_in.png" alt="ablation" width="620">
</p>

+ After fixing the bug:

<p align="center">
  <img src="../linear_separability/comparison_imagenet32_denoising_3546.png" alt="ablation" width="620">
</p>


> [!Note]
> - Now the PSNR at the final epoch aligns with that shown in Table 1, i.e., `22.082±0.012`
> - The gap between Hourglass and Hourglass (fix $W_{in}$) becomes larger (`22.076163 ± 0.002830` vs. `21.799570 ± 0.002052`), but still reasonablly small.

Increase training epochs:
+ Epoch = 2: `22.076163 ± 0.002830` vs. `21.799570 ± 0.002052`
+ Epoch = 4: `22.170157 ± 0.000246` vs. `21.922145 ± 0.002192`
+ Epoch = 8: same

Todo: How about Hourglass (fixed $W_{out}$)?

## 2.1 Case Study: MNIST Denoising
+ `data aug`: 1x
+ `batch size`:128, `epochs`: 100
+ `LR`: $\{5e-6, 1e-5, 5e-5, 1e-4, 5e-4\}$
<!-- + Report `mean ± std` over 5 runs -->
+ $d_{z}=784, d_{h}=1150$ 

<p align="center">
  <img src="./mnist_denoising_aug1/summary_best_train_loss_bs128_ep100_L_vs_metric.png" alt="train loss" width="280">
  <img src="./mnist_denoising_aug1/summary_best_eval_loss_bs128_ep100_L_vs_metric.png" alt="eval loss" width="280">
  <img src="./mnist_denoising_aug1/summary_test_psnr_bs128_ep100_L_vs_metric.png" alt="test psnr" width="280">
</p>


+ `data aug`: 2x
+ `batch size`:128, `epochs`: 100
+ `LR`: $\{5e-6, 1e-5, 5e-5, 1e-4, 5e-4\}$
<!-- + Report `mean ± std` over 5 runs -->
+ $d_{z}=784, d_{h}=1150$ 

<p align="center">
  <img src="./mnist_denoising_aug2/summary_best_train_loss_bs128_ep100_aug2_L_vs_metric.png" alt="train loss" width="280">
  <img src="./mnist_denoising_aug2/summary_best_eval_loss_bs128_ep100_aug2_L_vs_metric.png" alt="eval loss" width="280">
  <img src="./mnist_denoising_aug2/summary_test_psnr_bs128_ep100_aug2_L_vs_metric.png" alt="test psnr" width="280">
</p>

+ `data aug`: 4x
+ `batch size`:128, `epochs`: 100
+ `LR`: $\{5e-6, 1e-5, 5e-5, 1e-4, 5e-4\}$
<!-- + Report `mean ± std` over 5 runs -->
+ $d_{z}=784, d_{h}=1150$ 

<p align="center">
  <img src="./mnist_denoising_aug4/summary_best_train_loss_bs128_ep100_aug4_L_vs_metric.png" alt="train loss" width="280">
  <img src="./mnist_denoising_aug4/summary_best_eval_loss_bs128_ep100_aug4_L_vs_metric.png" alt="eval loss" width="280">
  <img src="./mnist_denoising_aug4/summary_test_psnr_bs128_ep100_aug4_L_vs_metric.png" alt="test psnr" width="280">
</p>
<P align="center">
Successfully prevent overfitting at L=6 and L=10. 
</p>

<!-- 
**Observations:**
- $W_{in}$ (Init N, Fix) performs worse than $w/o\ W_{in}$. This is not surprising given the debugged results from Section 1, where we showed that $W_{in}$ (Init N, Fix) cannot approximate $w/\ W_{in}$ as well as expected.
- For all kinds of $W_{in}$ variants, increasing $L$ beyond 4 starts to cause overfitting, evident by:
  - Lower `train_loss` ✓
  - Higher `eval_loss` ✗
- For $w/o\ W_{in}$, significantly more MLP layers are required to achieve comparable performance to $w/\ W_{in}$:
  - $w/o\ W_{in}$ with `L=15` ≈ $w/\ W_{in}$ with `L=6` (in terms of training loss)
  - This effect could be explained by the concept of `inductive bias` (e.g., prior knowledge on model architectures): 
    - [Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?](https://arxiv.org/abs/2207.10551)
    - [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
    - [Inductive Bias In Machine Learning | Medium](https://medium.com/@sanjithkumar986/inductive-bias-in-machine-learning-f360ea678a15)
- Note that most points in the figure are reported with `best_epoch=final_epoch=100`, indicating that the eval loss continued to decrease throughout training.
 -->

## 2.2 Case Study: MNIST Generative Classification
+ `data aug`: 1x
+ `batch size`:128, `epochs`: 100
+ `LR`: $\{5e-5, 1e-4, 5e-4, 5e-4\}$
<!-- + Report `mean ± std` over 5 runs -->
+ $d_{z}=784, d_{h}=1150$ 

<p align="center">
  <img src="./mnist_generative_classification/summary_best_train_loss_bs128_ep100_L_vs_metric.png" alt="train loss" width="280">
  <img src="./mnist_generative_classification/summary_best_eval_loss_bs128_ep100_L_vs_metric.png" alt="eval loss" width="280">
  <img src="./mnist_generative_classification/summary_test_psnr_bs128_ep100_L_vs_metric.png" alt="test psnr" width="280">
</p>


+ `data aug`: 2x
+ `batch size`:128, `epochs`: 100
+ `LR`: $\{5e-5, 1e-4, 5e-4, 5e-4\}$
<!-- + Report `mean ± std` over 5 runs -->
+ $d_{z}=784, d_{h}=1150$ 

<p align="center">
  <img src="./mnist_generative_classification_aug2/summary_best_train_loss_bs128_ep100_aug2_L_vs_metric.png" alt="train loss" width="280">
  <img src="./mnist_generative_classification_aug2/summary_best_eval_loss_bs128_ep100_aug2_L_vs_metric.png" alt="eval loss" width="280">
  <img src="./mnist_generative_classification_aug2/summary_test_psnr_bs128_ep100_aug2_L_vs_metric.png" alt="test psnr" width="280">
</p>


+ `data aug`: 4x
+ `batch size`:128, `epochs`: 100
+ `LR`: $\{5e-5, 1e-4, 5e-4, 5e-4\}$
<!-- + Report `mean ± std` over 5 runs -->
+ $d_{z}=784, d_{h}=1150$ 

<p align="center">
  <img src="./mnist_generative_classification_aug4/summary_best_train_loss_bs128_ep100_aug4_L_vs_metric.png" alt="train loss" width="280">
  <img src="./mnist_generative_classification_aug4/summary_best_eval_loss_bs128_ep100_aug4_L_vs_metric.png" alt="eval loss" width="280">
  <img src="./mnist_generative_classification_aug4/summary_test_psnr_bs128_ep100_aug4_L_vs_metric.png" alt="test psnr" width="280">
</p>

<!-- 
**Observations:**
- The performance ranking of different $W_{in}$ variants differs from Section 2.1
- All $w/\ W_{in}$ variants perform similarly and outperform $w/o\ W_{in}$
- $w/\ W_{in}$ and $w/o\ W_{in}$ achieve similar training loss, but $w/\ W_{in}$ has lower eval loss. This indicates that $w/o\ W_{in}$ fails to generalize well to unseen eval/test distributions -->


## 3.1 Case Study: ImageNet-32 Denoising

+ `data aug`: 1x
+ `batch size`:512, `epochs`: 2
+ `LR`: $\{1e-4, 3e-4\}$
<!-- + Report `mean ± std` over 2 runs -->
+ $d_{z}=3072, d_{h}=3546$

TBD

+ `data aug`: 4x
+ `batch size`:512, `epochs`: 2
+ `LR`: $\{1e-4, 3e-4\}$
<!-- + Report `mean ± std` over 2 runs -->
+ $d_{z}=3072, d_{h}=3546$

<p align="center">
  <img src="./imagenet32_denoising/summary_best_train_loss_bs512_ep2_aug4_L_vs_metric.png" alt="train loss" width="280">
  <img src="./imagenet32_denoising/summary_best_eval_loss_bs512_ep2_aug4_L_vs_metric.png" alt="eval loss" width="280">
  <img src="./imagenet32_denoising/summary_test_psnr_bs512_ep2_aug4_L_vs_metric.png" alt="test psnr" width="280">
</p>

<!-- 
+ `data aug`: 4x
+ `batch size`:512, `epochs`: 4
+ `LR`: $\{1e-4, 3e-4\}$ -> Need to try higer LR later
+ $d_{z}=3072, d_{h}=3546$

<p align="center">
  <img src="./imagenet32_denoising/summary_best_train_loss_bs512_ep4_aug4_L_vs_metric.png" alt="train loss" width="280">
  <img src="./imagenet32_denoising/summary_best_eval_loss_bs512_ep4_aug4_L_vs_metric.png" alt="eval loss" width="280">
  <img src="./imagenet32_denoising/summary_test_psnr_bs512_ep4_aug4_L_vs_metric.png" alt="test psnr" width="280">
</p> -->
