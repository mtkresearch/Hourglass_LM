# Rebuttal Preparation Plan (sorted by priority)

## 1. Ablation: Conventional MLP — w/ $W_{in}$ vs. w/o $W_{in}$

+ Ablation:
    1. Without $W_{in}$ → `Done`
    2. With $W_{in}$ → `Done`
        1. $W_{in}~N$ and trainable.
        3. $W_{in}=I$ and trainable.
        3. $W_{in}~N$ and fix.
        4. $W_{in}=I$ and fix. The performance should be the same as w/o $W_{in}$ → `True`
    3. Without $W_{out}$
        1. with $W_{in}$ (only do $W_{in}~N$ and trainable)
        2. witout $W_{in}$
        
    + Increase $L$ to see whether the performance gap decreases. 
    → `Not seeing this for now`: The performance of w/o $W_{in}$ plateau at a lower level.

    + Is the performance order the same across datasets and tasks?  → `No!`
        <!-- + `MNIST, Generative Classification` → `True`
        + `ImageNet, Denoising` → `False! Behave differently!` -->

+ Explain the results:
    + Calculate L2 norm of distance between inputs ($x$) and latents ($h$) after optimized $W_{in}$. Compared it with $h_{i+1} - h_{i}$.
    + ...

## 2. New Baseline: Conventional w/o both $W_{in}$ and $W_{out}$

+ Better wait for results from 1 before running grid search.

## 3. New Dataset: ImageNet (high resolution)

+ Just need to slightly modify the code for data processing.
+ Can run `hourglass`, but need to wait for 1 and 2 before running `conventional`.

## 4. Hourglass should include $W_{in} = W_{down}$

+ Surprisingly, for `MNIST-generative classification`, $W_{in} = W_{down}$ performs much better than $W_{in} = W_{up}$.

+ Check for other tasks, especially `ImageNet (high resolution)` implemented in 3 → We expect whether $W_{up}$ or $W_{down}$ is better is data- and task-dependent.
