import os
import copy
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchmetrics.functional import mean_squared_error
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

import torchvision.utils as vutils
from PIL import Image
import time


def save_samples_with_border(x, y_pred, y_star, img_shape, save_path, mode='autoencoder', down_scale=2.0):
    """Save sample images with border visualization for super resolution"""
    C, H, W = img_shape
    
    if mode == 'super':
        low_res_H, low_res_W = int(H // down_scale), int(W // down_scale)
        x_low_res = x.view(-1, C, low_res_H, low_res_W)
        batch_size = x_low_res.size(0)
        
        # Create images with border for better visualization
        x_with_border = torch.ones(batch_size, C, H, W) * 1.0
        pad_h, pad_w = (H - low_res_H) // 2, (W - low_res_W) // 2
        
        # Add black border and low res image
        border_width = 1
        x_with_border[:, :, pad_h-border_width:pad_h+low_res_H+border_width, 
                     pad_w-border_width:pad_w+low_res_W+border_width] = 0
        x_with_border[:, :, pad_h:pad_h+low_res_H, pad_w:pad_w+low_res_W] = x_low_res
        
        x_img = x_with_border
    else:
        x_img = x.view(-1, C, H, W)
    
    y_pred_img = y_pred.view(-1, C, H, W)
    y_star_img = y_star.view(-1, C, H, W)
    
    grid = torch.cat([x_img, y_pred_img, y_star_img], dim=0)
    grid = vutils.make_grid(grid, nrow=x.size(0), normalize=True, pad_value=1)
    ndarr = grid.mul(255).add_(0.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(ndarr).save(save_path)


@torch.no_grad()
def calculate_eval_loss_all_loaders(model, eval_loaders, device):
    """Calculate eval loss on all eval loaders"""
    model.eval()
    eval_losses = []
    
    for eval_loader in eval_loaders:
        total_loss, n_samples = 0, 0
        for x, y_star, _ in eval_loader:
            x, y_star = x.to(device), y_star.to(device)
            y_pred = model(x, y_star)
            total_loss += F.mse_loss(y_pred, y_star, reduction='mean').item() * x.size(0)
            n_samples += x.size(0)
        eval_losses.append(total_loss / n_samples)
    
    return eval_losses

@torch.no_grad()
def evaluate_overall_metrics(model, dataloader, device, img_shape):
    """Calculate overall PSNR and SSIM metrics using both individual and batch methods"""
    model.eval()
    C, H, W = img_shape
    batch_size = 128
    
    # Collect all data
    all_x, all_y_star = [], []
    for x, y_star, _ in dataloader:
        all_x.append(x)
        all_y_star.append(y_star)
    
    all_x = torch.cat(all_x, dim=0)
    all_y_star = torch.cat(all_y_star, dim=0)
    
    # # Individual-wise calculation
    # total_psnr, total_ssim, n_samples = 0, 0, 0
    # for start_idx in range(0, all_x.size(0), batch_size):
    #     end_idx = min(start_idx + batch_size, all_x.size(0))
    #     batch_x = all_x[start_idx:end_idx].to(device)
    #     batch_y_star = all_y_star[start_idx:end_idx].to(device)
        
    #     batch_y_pred = model(batch_x, batch_y_star)
    #     batch_y_pred_img = batch_y_pred.view(-1, C, H, W)
    #     batch_y_star_img = batch_y_star.view(-1, C, H, W)
        
    #     # Individual sample metrics
    #     for i in range(batch_x.size(0)):
    #         sample_pred = batch_y_pred_img[i:i+1]
    #         sample_star = batch_y_star_img[i:i+1]
    #         total_psnr += peak_signal_noise_ratio(sample_pred, sample_star, data_range=1.0).item()
    #         total_ssim += structural_similarity_index_measure(sample_pred, sample_star).item()
    #         n_samples += 1
    
    # psnr_overall = total_psnr / n_samples
    # ssim_overall = total_ssim / n_samples
    
    # Batch-wise calculation
    total_psnr_batch, total_ssim_batch, total_samples_batch = 0, 0, 0
    for start_idx in range(0, all_x.size(0), batch_size):
        end_idx = min(start_idx + batch_size, all_x.size(0))
        batch_x = all_x[start_idx:end_idx].to(device)
        batch_y_star = all_y_star[start_idx:end_idx].to(device)
        
        batch_y_pred = model(batch_x, batch_y_star)
        batch_y_pred_img = batch_y_pred.view(-1, C, H, W)
        batch_y_star_img = batch_y_star.view(-1, C, H, W)
        
        current_batch_size = batch_x.size(0)
        batch_psnr = peak_signal_noise_ratio(batch_y_pred_img, batch_y_star_img, data_range=1.0).item()
        batch_ssim = structural_similarity_index_measure(batch_y_pred_img, batch_y_star_img).item()
        
        total_psnr_batch += batch_psnr * current_batch_size
        total_ssim_batch += batch_ssim * current_batch_size
        total_samples_batch += current_batch_size
    
    psnr_batch_overall = total_psnr_batch / total_samples_batch
    ssim_batch_overall = total_ssim_batch / total_samples_batch
    
    return {
        'psnr': np.nan,  # 個別計算的整體指標暫不提供
        'ssim': np.nan,  # 個別計算的整體指標暫不提供
        'psnr_batch': psnr_batch_overall,
        'ssim_batch': ssim_batch_overall
    }


@torch.no_grad()
def evaluate_per_class_metrics(model, dataloader, device, img_shape, num_classes=10):
    """Calculate per-class PSNR and SSIM metrics using both individual and batch methods"""
    model.eval()
    C, H, W = img_shape
    batch_size = 128
    
    # Collect data by class
    class_data = {i: {'x': [], 'y_star': []} for i in range(num_classes)}
    
    for x, y_star, labels in dataloader:
        for i, label in enumerate(labels.tolist()):
            class_data[label]['x'].append(x[i])
            class_data[label]['y_star'].append(y_star[i])
    
    # Convert to tensors and calculate metrics
    class_metrics = {}
    for class_id in range(num_classes):
        if len(class_data[class_id]['x']) == 0:
            class_metrics[class_id] = {
                'psnr': 0, 'ssim': 0,
                'psnr_batch': 0, 'ssim_batch': 0,
                'count': 0
            }
            continue
            
        class_x = torch.stack(class_data[class_id]['x'])
        class_y_star = torch.stack(class_data[class_id]['y_star'])
        
        # Individual-wise calculation
        total_psnr, total_ssim, n_samples = 0, 0, 0
        for start_idx in range(0, class_x.size(0), batch_size):
            end_idx = min(start_idx + batch_size, class_x.size(0))
            batch_x = class_x[start_idx:end_idx].to(device)
            batch_y_star = class_y_star[start_idx:end_idx].to(device)
            
            batch_y_pred = model(batch_x, batch_y_star)
            batch_y_pred_img = batch_y_pred.view(-1, C, H, W)
            batch_y_star_img = batch_y_star.view(-1, C, H, W)
            
            for i in range(batch_x.size(0)):
                sample_pred = batch_y_pred_img[i:i+1]
                sample_star = batch_y_star_img[i:i+1]
                total_psnr += peak_signal_noise_ratio(sample_pred, sample_star, data_range=1.0).item()
                total_ssim += structural_similarity_index_measure(sample_pred, sample_star).item()
                n_samples += 1
        
        # Batch-wise calculation
        total_psnr_batch, total_ssim_batch, total_samples_batch = 0, 0, 0
        for start_idx in range(0, class_x.size(0), batch_size):
            end_idx = min(start_idx + batch_size, class_x.size(0))
            batch_x = class_x[start_idx:end_idx].to(device)
            batch_y_star = class_y_star[start_idx:end_idx].to(device)
            
            batch_y_pred = model(batch_x, batch_y_star)
            batch_y_pred_img = batch_y_pred.view(-1, C, H, W)
            batch_y_star_img = batch_y_star.view(-1, C, H, W)
            
            current_batch_size = batch_x.size(0)
            batch_psnr = peak_signal_noise_ratio(batch_y_pred_img, batch_y_star_img, data_range=1.0).item()
            batch_ssim = structural_similarity_index_measure(batch_y_pred_img, batch_y_star_img).item()
            
            total_psnr_batch += batch_psnr * current_batch_size
            total_ssim_batch += batch_ssim * current_batch_size
            total_samples_batch += current_batch_size
        
        class_metrics[class_id] = {
            'psnr': total_psnr / n_samples,
            'ssim': total_ssim / n_samples,
            'psnr_batch': total_psnr_batch / total_samples_batch,
            'ssim_batch': total_ssim_batch / total_samples_batch,
            'count': n_samples
        }
    
    return class_metrics


@torch.no_grad()
def final_test_evaluation(model, test_loaders, device, img_shape, track_per_class=True, num_classes=10):
    """Final comprehensive evaluation on all test loaders"""
    print(f"\n[INFO] Final test evaluation on {len(test_loaders)} test sets")
    
    all_test_results = []
    
    for i, test_loader in enumerate(test_loaders):
        print(f"      Evaluating test set {i+1}/{len(test_loaders)}")
        
        # Calculate overall metrics
        overall_metrics = evaluate_overall_metrics(model, test_loader, device, img_shape)
        
        # Calculate per-class metrics if needed
        class_metrics = None
        if track_per_class:
            class_metrics = evaluate_per_class_metrics(model, test_loader, device, img_shape, num_classes)
        
        all_test_results.append({
            'overall': overall_metrics,
            'per_class': class_metrics
        })
        
        # Display results
        print(f"        Test{i+1} Overall:")
        print(f"          PSNR: {overall_metrics['psnr']:.4f} | "
              f"PSNR_batch: {overall_metrics['psnr_batch']:.4f}")
        print(f"          SSIM: {overall_metrics['ssim']:.4f} | "
              f"SSIM_batch: {overall_metrics['ssim_batch']:.4f}")
    
    return all_test_results



def train_and_eval(model, 
                   train_loader, 
                   eval_loaders,
                   test_loaders,
                   img_shape, 
                   epochs, 
                   lr, 
                   device,
                   eval_interval,
                   primary_eval_idx=0,
                   model_name=None, 
                   exp_dir=None,
                   use_linear_decay=True,
                   min_lr=0.0,
                   track_per_class=False,
                   num_classes=10,
                   total_params=None, 
                   trainable_params=None, 
                   mode=None, 
                   down_scale=None,
                   use_step_logging=False,
                   log_interval_steps=50,
                   eval_interval_steps=500,
                   save_interval_steps=10000, 
                   use_warmup=False, 
                   warmup_ratio=0.1
                ):
    if not isinstance(eval_loaders, list):
        eval_loaders = [eval_loaders]
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    logging_mode = "step-based" if use_step_logging else "epoch-based"
    print(f"\n[INFO] Training {model_name} with {len(eval_loaders)} eval sets, {len(test_loaders)} test sets")
    print(f"[INFO] Using {logging_mode} logging")
    print(f"[INFO] Using eval set {primary_eval_idx} as primary for best model selection")

    if use_step_logging:
        print(f"[INFO] Step-based intervals: Log={log_interval_steps}, Eval={eval_interval_steps}, Save={save_interval_steps}")

    # ---- Optimizer / Scheduler ----
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = None

    if use_linear_decay:
        if use_step_logging:
            total_steps = epochs * len(train_loader)
            if use_warmup:
                warmup_steps = int(total_steps * warmup_ratio)

                def warmup_then_decay(step):
                    if step < warmup_steps:
                        # 緩啟動 10% -> 100%
                        return 0.1 + 0.9 * step / max(1, warmup_steps)
                    remaining_steps = max(1, total_steps - warmup_steps)
                    decay_step = step - warmup_steps
                    # 線性從 1.0 -> min_lr/lr
                    return 1.0 - (1.0 - min_lr / lr) * decay_step / remaining_steps

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_then_decay)
                print(f"[INFO] Using Warmup + Linear Decay: {lr} with {warmup_steps} warmup steps -> {min_lr} over {total_steps} steps")
            else:
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0, end_factor=min_lr / lr, total_iters=total_steps
                )
                print(f"[INFO] Using Linear Decay: {lr} -> {min_lr} over {total_steps} steps")
        else:
            # epoch-based
            if min_lr == 0.0:
                def linear_decay_to_zero(epoch):
                    return max(0.0, 1.0 - epoch / max(1, epochs))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_decay_to_zero)
                print(f"[INFO] Using Linear Decay: {lr} -> {min_lr} over {epochs} epochs (LambdaLR)")
            else:
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0, end_factor=min_lr / lr, total_iters=epochs
                )
                print(f"[INFO] Using Linear Decay: {lr} -> {min_lr} over {epochs} epochs (LinearLR)")
            if use_warmup:
                print("[WARNING] Warmup only supported with step-based logging, ignoring warmup for epoch-based training")
    else:
        print(f"[INFO] Using constant learning rate: {lr}")

    # ---- 目錄與樣本輸出 ----
    samples_dir = os.path.join(exp_dir, "samples") if exp_dir else None
    if samples_dir:
        os.makedirs(samples_dir, exist_ok=True)

    # ---- 紀錄容器 ----
    train_losses = []        # step-based: window avg；epoch-based: epoch avg
    eval_losses_all = []     # list of [eval0_loss, eval1_loss, ...]，在 step-based 的 log 週期或 epoch-based 每 epoch 儲存
    detailed_metrics = {'psnr': [], 'ssim': [], 'psnr_batch': [], 'ssim_batch': []}

    per_class_metrics = None
    if track_per_class:
        per_class_metrics = {
            'psnr': {i: [] for i in range(num_classes)},
            'ssim': {i: [] for i in range(num_classes)},
            'psnr_batch': {i: [] for i in range(num_classes)},
            'ssim_batch': {i: [] for i in range(num_classes)}
        }

    # ---- Best model tracking ----
    best_eval_loss = float('inf')
    best_model = None
    best_epoch = 0
    best_step = 0

    # ---- 計步器 / 時間器 ----
    global_step = 0
    last_log_time = time.time()

    # ---- 累積器（只用於 log 視窗平均的 train loss）----
    log_running_loss, log_running_steps = 0.0, 0

    # ---- 取一批 eval 資料做可視化（可選）----
    primary_eval_loader = eval_loaders[primary_eval_idx if 0 <= primary_eval_idx < len(eval_loaders) else 0]
    try:
        eval_batch_data = next(iter(primary_eval_loader))
        x_vis, y_star_vis = eval_batch_data[0].to(device), eval_batch_data[1].to(device)
    except Exception:
        x_vis, y_star_vis = None, None

    # =========================
    #        Training
    # =========================
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0

        # 取得當前 LR（step-based 會在每步後更新）
        current_lr = optimizer.param_groups[0]['lr']

        for batch_idx, (x, y_star, *rest) in enumerate(train_loader):
            x = x.to(device)
            y_star = y_star.to(device)

            optimizer.zero_grad(set_to_none=True) # 這邊不一樣
            y_pred = model(x, y_star)
            loss = F.mse_loss(y_pred, y_star)
            loss.backward()
            optimizer.step()

            batch_loss = float(loss.item())
            epoch_loss_sum += batch_loss
            global_step += 1

            # ---- 更新 log window 累積器 ----
            log_running_loss += batch_loss
            log_running_steps += 1

            # ---- Step-based scheduler ----
            if use_step_logging and scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

            # ---------- STEP-BASED 專屬：log / metrics / save ----------
            if use_step_logging:

                # A) Regular logging：每 log_interval_steps 計算/儲存 train_loss 與 eval_losses；做 best tracking
                if global_step % log_interval_steps == 0:
                    # 1) Train loss (視窗平均)
                    avg_train_loss = log_running_loss / max(1, log_running_steps)
                    train_losses.append(avg_train_loss)

                    # 2) Eval losses（所有 eval loaders）
                    eval_losses_step = calculate_eval_loss_all_loaders(model, eval_loaders, device)
                    eval_losses_all.append(eval_losses_step)

                    # 3) Best tracking（用 primary eval loss）
                    primary_idx = primary_eval_idx if 0 <= primary_eval_idx < len(eval_losses_step) else 0
                    primary_eval_loss = float(eval_losses_step[primary_idx])
                    if primary_eval_loss < best_eval_loss:
                        best_eval_loss = primary_eval_loss
                        best_epoch = epoch
                        best_step = global_step
                        best_model = copy.deepcopy(model)
                        print(f"      NEW BEST at Step {global_step}! Eval loss: {primary_eval_loss:.6f}")

                    # 4) 印出速度 / 概要
                    current_time = time.time()
                    time_per_step = (current_time - last_log_time) / max(1, log_interval_steps)
                    steps_per_sec = 1.0 / max(1e-9, time_per_step)
                    eval_losses_str = " | ".join([f"Eval{i}: {l:.4f}" for i, l in enumerate(eval_losses_step)])

                    print(f"[{model_name}] Step {global_step} | "
                          f"Epoch {epoch}/{epochs} ({batch_idx+1}/{len(train_loader)}) | "
                          f"LR: {current_lr:.6f} | "
                          f"Train(avg win): {avg_train_loss:.6f} | "
                          f"{eval_losses_str} | "
                          f"Speed: {steps_per_sec:.1f} steps/s | "
                          f"Time/step: {time_per_step:.3f}s | "
                          f"Params: {trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M "
                          f"({100*trainable_params/total_params:.1f}%)")

                    # 5) 重置 log 視窗累積器 / 計時器
                    log_running_loss, log_running_steps = 0.0, 0
                    last_log_time = current_time

                # B) Metrics interval：每 eval_interval_steps 只算/存 metrics（不算 eval losses）
                if global_step % eval_interval_steps == 0:
                    model.eval()
                    all_overall_metrics, all_class_metrics = [], []
                    for eval_idx, eval_loader in enumerate(eval_loaders):
                        om = evaluate_overall_metrics(model, eval_loader, device, img_shape)
                        all_overall_metrics.append(om)

                        cm = None
                        if track_per_class:
                            cm = evaluate_per_class_metrics(model, eval_loader, device, img_shape, num_classes)
                        all_class_metrics.append(cm)

                    # 儲存 overall metrics
                    detailed_metrics['psnr'].append([m['psnr'] for m in all_overall_metrics])
                    detailed_metrics['ssim'].append([m['ssim'] for m in all_overall_metrics])
                    detailed_metrics['psnr_batch'].append([m['psnr_batch'] for m in all_overall_metrics])
                    detailed_metrics['ssim_batch'].append([m['ssim_batch'] for m in all_overall_metrics])

                    # 儲存 per-class metrics
                    if track_per_class:
                        for class_id in range(num_classes):
                            per_class_metrics['psnr'][class_id].append([
                                cm[class_id]['psnr'] if cm is not None else float('nan')
                                for cm in all_class_metrics
                            ])
                            per_class_metrics['ssim'][class_id].append([
                                cm[class_id]['ssim'] if cm is not None else float('nan')
                                for cm in all_class_metrics
                            ])
                            per_class_metrics['psnr_batch'][class_id].append([
                                cm[class_id]['psnr_batch'] if cm is not None else float('nan')
                                for cm in all_class_metrics
                            ])
                            per_class_metrics['ssim_batch'][class_id].append([
                                cm[class_id]['ssim_batch'] if cm is not None else float('nan')
                                for cm in all_class_metrics
                            ])

                    # 印主要 eval 的 metrics
                    primary_idx = primary_eval_idx if 0 <= primary_eval_idx < len(all_overall_metrics) else 0
                    pm = all_overall_metrics[primary_idx]
                    psnr_val, ssim_val = pm['psnr'], pm['ssim']
                    psnr_batch_val, ssim_batch_val = pm['psnr_batch'], pm['ssim_batch']
                    psnr_str = f"{psnr_val:.4f}" if not np.isnan(psnr_val) else "----"
                    ssim_str = f"{ssim_val:.4f}" if not np.isnan(ssim_val) else "----"
                    psnr_batch_str = f"{psnr_batch_val:.4f}" if not np.isnan(psnr_batch_val) else "----"
                    ssim_batch_str = f"{ssim_batch_val:.4f}" if not np.isnan(ssim_batch_val) else "----"

                    print(f"[{model_name}] Step {global_step} | "
                          f"Epoch {epoch}/{epochs} ({batch_idx+1}/{len(train_loader)}) | "
                          f"LR: {current_lr:.6f} | "
                          f"PSNR: {psnr_str} | "
                          f"PSNR_batch: {psnr_batch_str} | "
                          f"SSIM: {ssim_str} | "
                          f"SSIM_batch: {ssim_batch_str}")

                    model.train()

                # C) 依步數存樣本或 checkpoint（可選）
                if save_interval_steps and samples_dir and (global_step % save_interval_steps == 0) and (x_vis is not None):
                    with torch.no_grad():
                        model.eval()
                        y_pred_vis = model(x_vis, y_star_vis)
                        save_path = os.path.join(samples_dir, f"step_{global_step:07d}.png")
                        try:
                            save_samples_with_border(x_vis, y_pred_vis, y_star_vis, img_shape, save_path, mode=mode, down_scale=down_scale)
                        except Exception as e:
                            print(f"[WARN] save_samples_with_border failed at step {global_step}: {e}")
                        model.train()

        # ---------- EPOCH-BASED 專屬：每個 epoch 統計與評估 ----------
        if not use_step_logging:
            # Train loss（本 epoch 平均）
            avg_train_loss_epoch = epoch_loss_sum / max(1, len(train_loader))
            train_losses.append(avg_train_loss_epoch)

            # Eval losses（所有 eval loaders）
            eval_losses_epoch = calculate_eval_loss_all_loaders(model, eval_loaders, device)
            eval_losses_all.append(eval_losses_epoch)

            # Best tracking（以 primary eval loss）
            primary_idx = primary_eval_idx if 0 <= primary_eval_idx < len(eval_losses_epoch) else 0
            primary_eval_loss = float(eval_losses_epoch[primary_idx])
            if primary_eval_loss < best_eval_loss:
                best_eval_loss = primary_eval_loss
                best_epoch = epoch
                best_step = global_step
                best_model = copy.deepcopy(model)
                print(f"      NEW BEST at Epoch {epoch}! Eval loss: {primary_eval_loss:.6f}")

            # Metrics（每個 epoch 都算一次）
            model.eval()
            all_overall_metrics, all_class_metrics = [], []
            for eval_idx, eval_loader in enumerate(eval_loaders):
                om = evaluate_overall_metrics(model, eval_loader, device, img_shape)
                all_overall_metrics.append(om)

                cm = None
                if track_per_class:
                    cm = evaluate_per_class_metrics(model, eval_loader, device, img_shape, num_classes)
                all_class_metrics.append(cm)

            detailed_metrics['psnr'].append([m['psnr'] for m in all_overall_metrics])
            detailed_metrics['ssim'].append([m['ssim'] for m in all_overall_metrics])
            detailed_metrics['psnr_batch'].append([m['psnr_batch'] for m in all_overall_metrics])
            detailed_metrics['ssim_batch'].append([m['ssim_batch'] for m in all_overall_metrics])

            if track_per_class:
                for class_id in range(num_classes):
                    per_class_metrics['psnr'][class_id].append([
                        cm[class_id]['psnr'] if cm is not None else float('nan')
                        for cm in all_class_metrics
                    ])
                    per_class_metrics['ssim'][class_id].append([
                        cm[class_id]['ssim'] if cm is not None else float('nan')
                        for cm in all_class_metrics
                    ])
                    per_class_metrics['psnr_batch'][class_id].append([
                        cm[class_id]['psnr_batch'] if cm is not None else float('nan')
                        for cm in all_class_metrics
                    ])
                    per_class_metrics['ssim_batch'][class_id].append([
                        cm[class_id]['ssim_batch'] if cm is not None else float('nan')
                        for cm in all_class_metrics
                    ])
            model.train()

            # Epoch-based scheduler（每個 epoch 結束後 step 一次）
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

        # ---- 每個 epoch 結尾的樣本可視化（可選）----
        if (not use_step_logging) and samples_dir and (x_vis is not None):
            with torch.no_grad():
                model.eval()
                y_pred_vis = model(x_vis, y_star_vis)
                save_path = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
                try:
                    save_samples_with_border(x_vis, y_pred_vis, y_star_vis, img_shape, save_path, mode=mode, down_scale=down_scale)
                except Exception as e:
                    print(f"[WARN] save_samples_with_border failed at epoch {epoch}: {e}")
                model.train()

    # =========================
    #       Finalization
    # =========================
    # 若沒有任何 best（極少數情況），回退到當前模型
    if best_model is None:
        best_model = copy.deepcopy(model)

    # 最終在 test sets 上評估（可選）
    final_test_metrics = []
    try:
        model.eval()
        for t_idx, test_loader in enumerate(test_loaders):
            om = evaluate_overall_metrics(best_model, test_loader, device, img_shape)
            final_test_metrics.append(om)
    except Exception as e:
        print(f"[WARN] final test evaluation skipped: {e}")

    summary = {
        "train_losses": train_losses,
        "eval_losses_all": eval_losses_all,         # list of lists；step-based: 在 log 週期更新
        "detailed_metrics": detailed_metrics,       # 收集於 metrics 週期（step-based）或每 epoch（epoch-based）
        "per_class_metrics": per_class_metrics,
        "best": {
            "eval_loss": best_eval_loss,
            "epoch": best_epoch,
            "step": best_step,
        },
        "final_test_metrics": final_test_metrics,
        "meta": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lr": lr,
            "min_lr": min_lr,
            "use_step_logging": use_step_logging,
            "log_interval_steps": log_interval_steps,
            "eval_interval_steps": eval_interval_steps,
        }
    }

    # 回傳 best_model（權重）與訓練摘要
    return best_model, summary
