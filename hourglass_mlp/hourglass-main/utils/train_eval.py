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

def save_samples_with_border(x, y_pred, y_gt, img_shape, save_path, mode, down_scale):
    """Save sample images with border visualization for super resolution"""
    C, H, W = img_shape
    device = x.device
    dtype = x.dtype

    if mode == 'super_resolution':
        low_res_H, low_res_W = int(H // down_scale), int(W // down_scale)
        x_low_res = x.view(-1, C, low_res_H, low_res_W)  # already on device
        batch_size = x_low_res.size(0)

        # create canvas on the SAME device/dtype as x
        x_with_border = torch.ones(batch_size, C, H, W, device=device, dtype=dtype)

        pad_h, pad_w = (H - low_res_H) // 2, (W - low_res_W) // 2
        border_width = 1

        # black border (also on same device)
        x_with_border[:, :, pad_h-border_width:pad_h+low_res_H+border_width,
                             pad_w-border_width:pad_w+low_res_W+border_width] = 0

        # paste low-res in the center
        x_with_border[:, :, pad_h:pad_h+low_res_H, pad_w:pad_w+low_res_W] = x_low_res

        x_img = x_with_border
    else:
        x_img = x.view(-1, C, H, W)
        batch_size = x_img.size(0)

    y_pred_img = y_pred.view(-1, C, H, W).to(device=device, dtype=dtype)
    y_star_img = y_gt.view(-1, C, H, W).to(device=device, dtype=dtype)
    grid = torch.cat([x_img, y_pred_img, y_star_img], dim=0)
    grid = vutils.make_grid(grid, nrow=batch_size, normalize=True, pad_value=1.0)

    ndarr = (
        grid.detach().cpu().mul(255).add_(0.5).clamp(0, 255)
        .byte().permute(1, 2, 0).numpy()
    )
    Image.fromarray(ndarr).save(save_path)

@torch.no_grad()
def _eval_loss_psnr_and_ssim(model, loader, device, img_shape, 
                              viz_path=None, mode=None, down_scale=None):
    """Evaluate MSE loss, PSNR, and SSIM on a single dataloader."""
    model.eval()
    C, H, W = img_shape
    total_loss, total_psnr, total_ssim, total_cnt = 0.0, 0.0, 0.0, 0
    viz_saved = False

    for x, y_gt in loader:
        x = x.to(device)
        y_gt = y_gt.to(device)

        y_pred = model(x)
        loss = F.mse_loss(y_pred, y_gt, reduction='mean')
        bsz = x.size(0)

        total_loss += loss.item() * bsz

        y_pred_img = y_pred.view(-1, C, H, W)
        y_gt_img   = y_gt.view(-1, C, H, W)
        batch_psnr = peak_signal_noise_ratio(y_pred_img, y_gt_img, data_range=1.0).item()
        batch_ssim = structural_similarity_index_measure(y_pred_img, y_gt_img).item()
        total_psnr += batch_psnr * bsz
        total_ssim += batch_ssim * bsz
        total_cnt  += bsz

        # save visualization for first 20 images in the first batch
        if (viz_path is not None) and (not viz_saved):
            k = min(20, bsz)
            x20      = x[:k].detach()
            y_pred20 = y_pred[:k].detach()
            y_gt20   = y_gt[:k].detach()
            save_samples_with_border(x20, y_pred20, y_gt20, img_shape, viz_path, mode, down_scale)
            viz_saved = True

    avg_loss = total_loss / total_cnt
    avg_psnr = total_psnr / total_cnt
    avg_ssim = total_ssim / total_cnt
    return avg_loss, avg_psnr, avg_ssim


def train_and_eval(model, model_type, trainable_params, total_params, 
                   train_loader, eval_loader, test_loader,
                   img_shape, save_path, mode, down_scale,
                   device,
                   epochs, lr, min_lr=0.0,
                   logging_step='epoch_based',   # or 'step_based'
                   eval_interval_epochs=50,
                   log_interval_steps=500,
                   eval_interval_steps=5000,
                   ):
    
    assert logging_step in ('epoch_based', 'step_based'), \
        f"logging_step must be 'epoch_based' or 'step_based', got {logging_step!r}"

    os.makedirs(os.path.join(save_path, 'visualization'), exist_ok=True)

    print(f"\n[INFO] Training {model_type} MLP")
    print(f"[INFO] Logging mode: {logging_step}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    # Scheduler (Linear decay)
    if logging_step == 'step_based':
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=(min_lr / lr if lr > 0 else 0.0),
            total_iters=max(1, total_steps),
        )
        print(f"[INFO] LinearLR (step-based): {lr} -> {min_lr} over {total_steps} steps")
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=(min_lr / lr if lr > 0 else 0.0),
            total_iters=max(1, epochs),
        )
        print(f"[INFO] Linear LR Decay (epoch-based): {lr} -> {min_lr} over {epochs} epochs")

    # Storage
    train_losses = []
    eval_losses  = []
    psnr_history = []
    ssim_history = []

    best_eval_loss = float('inf')
    best_epoch, best_step = 0, 0
    best_model = copy.deepcopy(model)

    global_step = 0
    running_loss = 0.0
    last_log_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_cnt = 0

        for batch_idx, (x, y_gt) in enumerate(train_loader, start=1):
            x = x.to(device)
            y_gt = y_gt.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y_gt, reduction='mean')
            loss.backward()
            optimizer.step()

            # accounting
            bsz = x.size(0)
            epoch_loss_sum += loss.item() * bsz
            epoch_cnt += bsz
            running_loss += loss.item()
            global_step += 1

            # step-based logging, lr, eval
            if logging_step == 'step_based':
                scheduler.step()

                if global_step % log_interval_steps == 0:
                    now = time.time()
                    time_per_step = (now - last_log_time) / log_interval_steps
                    steps_per_sec = 1.0 / max(1e-12, time_per_step)
                    avg_loss_chunk = running_loss / log_interval_steps
                    train_losses.append(avg_loss_chunk)
                    curr_lr = optimizer.param_groups[0]['lr']
                    print(f"[{model_type} ({trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M)] Step: {global_step} | Epoch: {epoch}/{epochs} "
                          f"| LR: {curr_lr:.6f} | Train Loss: {avg_loss_chunk:.6f} | {steps_per_sec:.1f} steps/s")
                    running_loss = 0.0
                    last_log_time = now

                if global_step % eval_interval_steps == 0:
                    curr_viz_path = os.path.join(save_path, 'visualization', f'step{global_step}.png')
                    loss_val, psnr_val, ssim_val = _eval_loss_psnr_and_ssim(model, eval_loader, device, img_shape, 
                                                                             curr_viz_path, mode, down_scale)    
                    eval_losses.append(loss_val)
                    psnr_history.append(psnr_val)
                    ssim_history.append(ssim_val)

                    if loss_val < best_eval_loss:
                        best_eval_loss = loss_val
                        best_epoch = epoch
                        best_step = global_step
                        best_model = copy.deepcopy(model)
                        print(f"      [BEST @ step {global_step}] eval loss: {loss_val:.6f}")

                    model.train()  # back to train mode

        # epoch-based logging, lr, eval
        if logging_step == 'epoch_based':
            epoch_avg_loss = epoch_loss_sum / epoch_cnt
            train_losses.append(epoch_avg_loss)
            scheduler.step()

            if (epoch % eval_interval_epochs == 0) or (epoch == epochs):
                curr_viz_path = os.path.join(save_path, 'visualization', f'epoch{epoch}.png')
                loss_val, psnr_val, ssim_val = _eval_loss_psnr_and_ssim(model, eval_loader, device, img_shape, 
                                                                         curr_viz_path, mode, down_scale)    
                eval_losses.append(loss_val)
                psnr_history.append(psnr_val)
                ssim_history.append(ssim_val)

                if loss_val < best_eval_loss:
                    best_eval_loss = loss_val
                    best_epoch = epoch
                    best_step = global_step
                    best_model = copy.deepcopy(model)
                    print(f"      [BEST @ epoch {epoch}] eval loss: {loss_val:.6f}")

                curr_lr = optimizer.param_groups[0]['lr']
                print(f"[{model_type} ({trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M)] Epoch: {epoch}/{epochs} | LR: {curr_lr:.6f} | "
                      f"Train Loss: {epoch_avg_loss:.6f} | Eval Loss: {loss_val:.6f} | PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}")

    # Final testing with best model
    test_loss, test_psnr, test_ssim = _eval_loss_psnr_and_ssim(best_model, test_loader, device, img_shape)
    if logging_step == 'step_based':
        print(f"\n[INFO] Using best model @ step: {best_step}, test loss: {test_loss:.6f} | PSNR: {test_psnr:.4f} | SSIM: {test_ssim:.4f}")
    else:
        print(f"\n[INFO] Using best model @ epoch {best_epoch}, test loss: {test_loss:.6f} | PSNR: {test_psnr:.4f} | SSIM: {test_ssim:.4f}")

    training_logs = {
        "train_losses": train_losses,
        "eval_losses":  eval_losses,
        "psnr":         psnr_history,
        "ssim":         ssim_history,
        "best_epoch":   (best_epoch if logging_step == 'epoch_based' else None),
        "best_step":    (best_step if logging_step == 'step_based' else None),
        "best_eval_loss": best_eval_loss,
    }

    test_results = {
        "test_loss": test_loss,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim,
    }

    return training_logs, test_results