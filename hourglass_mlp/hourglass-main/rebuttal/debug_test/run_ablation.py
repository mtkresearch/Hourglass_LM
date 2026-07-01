import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json, random, argparse
from PIL import Image
from thop import profile
from tabulate import tabulate


from src.utils.ds import PairDataset
from src.utils.model import HourGlassMLP, NormalMLP
from src.utils.flops import get_model_stats, filter_flops_info
# from src.utils.train_eval import train_and_eval
from src.utils.train_eval_for_ablation import train_and_eval

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp_home', type=str, default='./results')
    
    parser.add_argument('--ds_name', type=str, default='mnist', choices=['mnist', 'cifar100', 'bsds500', 'imagenet32'])
    parser.add_argument('--mode', type=str, default='class_to_prototype', choices=['autoencoder', 'denoising', 'class_to_prototype', 'super'])
    parser.add_argument('--noise_std', type=float, default=0.25)
    # parser.add_argument('--down_scale', type=float, default=2.0)
    # parser.add_argument('--down_scale', type=float, default=4.0)
    parser.add_argument('--down_scale', type=float)
    
    # choose which model to run
    parser.add_argument('--model_type', type=str, default='hourglass', choices=['hourglass', 'normal'])
        
    # model architecture
    parser.add_argument('--reps_dim', type=int, default=1024)
    parser.add_argument('--mid_dims', nargs='+', type=int, required=True, help='List of middle dims')
    parser.add_argument('--normal_in_proj', action='store_true', default=False,
                    help='Enable input projection')
    parser.add_argument('--normal_out_proj', action='store_true', default=False,
                    help='Enable output projection')
    
    # training hyperparmeters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_interval', type=int, default=1000)   
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                    help='Enable data augmentation')
    parser.add_argument('--aug_num', type=int, default=2, choices=[2, 4])
    parser.add_argument('--use_scaling_aug', action='store_true', default=False,
                    help='Enable **online** scaling data augmentation')
    


    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    
    ## For hourglass
    parser.add_argument('--freeze_hg_in_out', action='store_true', default=False,
                    help='Do not train the input/output of hourglass')
    
    return parser.parse_args()

def set_seed(seed):
    '''
    set torch random seed
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Basic setup
    args = parse_arguments()
    device = args.device
    print(device)
   
    if args.ds_name in ['mnist']:
        input_dim = 784
        output_dim = 784
        num_classes = 10       
        use_step_logging = False  
    elif args.ds_name in ['cifar100']:
        input_dim = 3072
        output_dim = 3072
        num_classes = 100
        use_step_logging = False  
    elif args.ds_name in ['imagenet32']:
        input_dim = 3072
        output_dim = 3072
        num_classes = None
        use_step_logging = True
        args.down_scale = 2.0
        # args.down_scale = 4.0
    elif args.ds_name in ['bsds500']:
        input_dim = 4096
        output_dim = 4096
        num_classes = None     
        use_step_logging = False  
    else:
        raise NotImplementedError()
    
    if args.mode == 'super':
        input_dim = int(input_dim // (args.down_scale) ** 2)
        

    if args.model_type == 'hourglass':
        model = HourGlassMLP(input_dim=input_dim, 
                             output_dim=output_dim,
                             reps_dim=args.reps_dim, 
                             mid_dims=args.mid_dims,
                             compact_mid=False).to(device)
        
        if args.freeze_hg_in_out:
            print('Freezing input of Hourglass model')            # 直接對整個模塊凍結
            model.up.requires_grad_(False)

            print('Freezing output of Hourglass model')            # 直接對整個模塊凍結
            # model.down.requires_grad_(False)
            
            # 驗證凍結狀態
            print(f"Up layer frozen: {not any(p.requires_grad for p in model.up.parameters())}")
            print(f"Down layer frozen: {not any(p.requires_grad for p in model.down.parameters())}")

    elif args.model_type == 'normal':
        if not(args.normal_in_proj == args.normal_out_proj):
            raise ValueError('currently we only suppoprt adding or disabling in/out proj at the same time!')
        print(f'Initialize NormalMLP with [input projection: {args.normal_in_proj} | output projection:{args.normal_out_proj}]')
        model = NormalMLP(input_dim=input_dim, 
                          output_dim=output_dim,
                          reps_dim=args.reps_dim, 
                          mid_dims=args.mid_dims, 
                          in_proj=args.normal_in_proj, out_proj=args.normal_out_proj).to(device)
    print(model)
    total_params, total_flops, trainable_params, skipadd_ratio, flops_per_block = get_model_stats(model, device, input_dim, output_dim)
    model_stats = filter_flops_info(total_params, total_flops, trainable_params, skipadd_ratio, flops_per_block)

    # Output FLOPs table
    table_data = [[
        args.model_type,
        model_stats["total_params"],
        model_stats["trainable_params"],
        model_stats["total_flops"],
        f"{model_stats['skipadd_flops_ratio']*100:.2f}%"
    ]]
    print("\n[Model Stats]")
    print(tabulate(table_data, headers=["Model", "Total Params", "Trainable Params", "Total FLOPs", "SkipAdd FLOPs Ratio"]))

    # Setup exp name
    mid_dims_str = '_'.join(map(str, args.mid_dims))
    if args.model_type == 'normal':
        if args.normal_in_proj and args.normal_out_proj:
            model_type_str = 'normal_w_in_out'
        elif args.normal_in_proj:
            model_type_str = 'normal_w_in'
        elif args.normal_out_proj:
            model_type_str = 'normal_w_out'
        else:
            model_type_str = 'normal'
    elif args.model_type == 'hourglass':
        if args.freeze_hg_in_out:
            # model_type_str = 'hourglass_freeze_inout'
            model_type_str = 'hourglass_freeze_in'
        else:
            model_type_str = 'hourglass'


    if args.mode == 'denoising':
        if args.use_augmentation:
            exp_dir = f"{args.exp_home}_bs{args.batch_size}_ep{args.epochs}/{args.ds_name}_{args.mode}_std{args.noise_std}_aug{args.aug_num}/{trainable_params}/{model_type_str}/reps{args.reps_dim}_mid{mid_dims_str}/lr{args.lr}_run{args.run_id}"
        else:
            exp_dir = f"{args.exp_home}_bs{args.batch_size}_ep{args.epochs}/{args.ds_name}_{args.mode}_std{args.noise_std}/{trainable_params}/{model_type_str}/reps{args.reps_dim}_mid{mid_dims_str}/lr{args.lr}_run{args.run_id}"
    elif args.mode == 'super':
        if args.use_augmentation and not args.use_scaling_aug:
            exp_dir = f"{args.exp_home}_bs{args.batch_size}_ep{args.epochs}/{args.ds_name}_{args.mode}_down{args.down_scale}_aug{args.aug_num}/{trainable_params}/{model_type_str}/reps{args.reps_dim}_mid{mid_dims_str}/lr{args.lr}_run{args.run_id}"
        elif args.use_augmentation and args.use_scaling_aug:
            exp_dir = f"{args.exp_home}_bs{args.batch_size}_ep{args.epochs}/{args.ds_name}_{args.mode}_down{args.down_scale}_aug{args.aug_num}_scale/{trainable_params}/{model_type_str}/reps{args.reps_dim}_mid{mid_dims_str}/lr{args.lr}_run{args.run_id}"
        
        else:
            exp_dir = f"{args.exp_home}_bs{args.batch_size}_ep{args.epochs}/{args.ds_name}_{args.mode}_down{args.down_scale}/{trainable_params}/{model_type_str}/reps{args.reps_dim}_mid{mid_dims_str}/lr{args.lr}_run{args.run_id}"
    else:
        if args.use_augmentation:
            raise NotImplementedError('currently we only support data augmentation for denoising and super-resolution tasks!')
        exp_dir = f"{args.exp_home}_bs{args.batch_size}_ep{args.epochs}/{args.ds_name}_{args.mode}/{trainable_params}/{model_type_str}/reps{args.reps_dim}_mid{mid_dims_str}/lr{args.lr}_run{args.run_id}"

    # If the exp dir already exist, return error
    if os.path.exists(exp_dir):
        raise ValueError(f"Dupicated experiment: [{exp_dir}] already exist!")
    else:
        os.makedirs(exp_dir, exist_ok=True)
    
    #region [data setup]
    ## Create base dataset to get some info
    if args.mode == 'class_to_prototype':
        base_dataset = PairDataset(args.ds_name, mode=args.mode, split='train', noise_std=args.noise_std, down_scale=args.down_scale)
        shared_prototypes = base_dataset.prototypes if args.mode == 'class_to_prototype' else None
    
    ## Load train, val, and test set
    train_set = PairDataset(args.ds_name, mode=args.mode, split='train', 
                            noise_std=args.noise_std, down_scale=args.down_scale,
                            use_augmentation=args.use_augmentation, aug_num=args.aug_num, use_scaling_aug=args.use_scaling_aug)
    img_shape = (train_set.C, train_set.H, train_set.W)

    ## Create data loader 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    if args.mode == 'denoising':
        print('preparing noisy eval/test data, std=0.1')
        val_set_1 = PairDataset(args.ds_name, mode=args.mode, split='eval', noise_std=0.1, down_scale=args.down_scale, use_augmentation=False)
        test_set_1 = PairDataset(args.ds_name, mode=args.mode, split='test', noise_std=0.1, down_scale=args.down_scale, use_augmentation=False)

        print('preparing noisy eval/test data, std=0.25')
        val_set_2 = PairDataset(args.ds_name, mode=args.mode, split='eval', noise_std=0.25, down_scale=args.down_scale, use_augmentation=False)
        test_set_2 = PairDataset(args.ds_name, mode=args.mode, split='test', noise_std=0.25, down_scale=args.down_scale, use_augmentation=False)
        
        print('preparing noisy eval/test data, std=0.50')
        val_set_3 = PairDataset(args.ds_name, mode=args.mode, split='eval', noise_std=0.50, down_scale=args.down_scale, use_augmentation=False)
        test_set_3 = PairDataset(args.ds_name, mode=args.mode, split='test', noise_std=0.50, down_scale=args.down_scale, use_augmentation=False)
        
        val_loaders = [
            DataLoader(val_set_1, batch_size=args.batch_size, shuffle=False),
            DataLoader(val_set_2, batch_size=args.batch_size, shuffle=False),
            DataLoader(val_set_3, batch_size=args.batch_size, shuffle=False)
        ]
        test_loaders = [
            DataLoader(test_set_1, batch_size=args.batch_size, shuffle=False),
            DataLoader(test_set_2, batch_size=args.batch_size, shuffle=False),
            DataLoader(test_set_3, batch_size=args.batch_size, shuffle=False)
        ]
        primary_eval_idx = 1
    else:
        val_set = PairDataset(args.ds_name, mode=args.mode, split='eval', noise_std=args.noise_std, down_scale=args.down_scale, use_augmentation=False)
        test_set = PairDataset(args.ds_name, mode=args.mode, split='test', noise_std=args.noise_std, down_scale=args.down_scale, use_augmentation=False)
        
        val_loaders = [DataLoader(val_set, batch_size=args.batch_size, shuffle=False)]
        test_loaders = [DataLoader(test_set, batch_size=args.batch_size, shuffle=False)]
        primary_eval_idx = 0

    if args.mode == 'class_to_prototype' and shared_prototypes is not None:
        train_set.prototypes = shared_prototypes
        for val_loader in val_loaders:
            val_loader.dataset.prototypes = shared_prototypes
        for test_loader in test_loaders:
            test_loader.dataset.prototypes = shared_prototypes

    
    ## Save ground truth images
    gt_image_paths = {}
    if args.mode == 'class_to_prototype':
        gt_dir = os.path.join(exp_dir, "gt_images")
        os.makedirs(gt_dir, exist_ok=True)
        for label, img in shared_prototypes.items():
            img_path = os.path.join(gt_dir, f"gt_{label}.png")
            ndarr = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
            if img.shape[0] == 1:
                Image.fromarray(ndarr.squeeze(), mode='L').save(img_path)
            else:
                Image.fromarray(ndarr, mode='RGB').save(img_path)
            gt_image_paths[label] = img_path
    #endregion

    #region [training]
    set_seed(args.seed + args.run_id) # set different seed for different run id        

    # Only generative classification need to separate calculate metrics
    if args.mode == 'class_to_prototype':
        track_per_class = True
    else:
        track_per_class = False

    best_model, training_summary = train_and_eval(
        model, 
        train_loader, 
        val_loaders, 
        test_loaders,
        img_shape,
        args.epochs, 
        args.lr, 
        device,
        args.eval_interval,
        primary_eval_idx=primary_eval_idx,
        model_name=args.model_type, 
        exp_dir=exp_dir,
        track_per_class=track_per_class,
        num_classes=num_classes, 
        total_params=total_params, 
        trainable_params=trainable_params, 
        mode=args.mode, 
        down_scale=args.down_scale, 
        use_step_logging=use_step_logging
    )

    # ----------------- 存檔區塊 -----------------
    for eval_idx in range(len(val_loaders)):
        if eval_idx == 0:
            summary_filename = "experiment_summary.json"
        else:
            summary_filename = f"experiment_summary_eval{eval_idx + 1}.json"

        overall_metrics = {
            'train_loss': training_summary['train_losses'],  # ← 用 training_summary
            'eval_loss': [epoch_losses[eval_idx] for epoch_losses in training_summary['eval_losses_all']],
            'psnr': [epoch_metrics[eval_idx] if isinstance(epoch_metrics, list) else epoch_metrics 
                            for epoch_metrics in training_summary['detailed_metrics']['psnr']],
            'psnr_batch': [epoch_metrics[eval_idx] if isinstance(epoch_metrics, list) else epoch_metrics 
                                for epoch_metrics in training_summary['detailed_metrics']['psnr_batch']],
            'ssim': [epoch_metrics[eval_idx] if isinstance(epoch_metrics, list) else epoch_metrics 
                            for epoch_metrics in training_summary['detailed_metrics']['ssim']],
            'ssim_batch': [epoch_metrics[eval_idx] if isinstance(epoch_metrics, list) else epoch_metrics 
                                for epoch_metrics in training_summary['detailed_metrics']['ssim_batch']],
        }

        current_per_class_metrics = None
        if track_per_class and training_summary['per_class_metrics']:
            current_per_class_metrics = {}
            for class_id in range(num_classes):
                current_per_class_metrics[class_id] = {
                    'psnr': [epoch_metrics[eval_idx] if isinstance(epoch_metrics, list) else epoch_metrics 
                                    for epoch_metrics in training_summary['per_class_metrics']['psnr'][class_id]],
                    'psnr_batch': [epoch_metrics[eval_idx] if isinstance(epoch_metrics, list) else epoch_metrics 
                                        for epoch_metrics in training_summary['per_class_metrics']['psnr_batch'][class_id]],
                    'ssim': [epoch_metrics[eval_idx] if isinstance(epoch_metrics, list) else epoch_metrics 
                                    for epoch_metrics in training_summary['per_class_metrics']['ssim'][class_id]],
                    'ssim_batch': [epoch_metrics[eval_idx] if isinstance(epoch_metrics, list) else epoch_metrics 
                                        for epoch_metrics in training_summary['per_class_metrics']['ssim_batch'][class_id]],
                }

        # 「test_results」來源調整：新版在 summary['final_test_metrics']
        test_results = training_summary['final_test_metrics'][eval_idx] if eval_idx < len(training_summary['final_test_metrics']) else None

        summary = {
            "config": vars(args),
            "gt_image_paths": gt_image_paths,
            "model_stats": model_stats,
            "overall_metrics": overall_metrics,
            "eval_loader_index": eval_idx,
            "best_epoch": training_summary['best']['epoch'],        # ← 改這裡
            "best_eval_loss": training_summary['best']['eval_loss'],# ← 改這裡
            "test_results": test_results
        }
        if current_per_class_metrics:
            summary["per_class_metrics"] = current_per_class_metrics

        summary_path = os.path.join(exp_dir, summary_filename)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Saved eval loader {eval_idx + 1} results to {summary_filename}")

if __name__ == '__main__':
    main() 
