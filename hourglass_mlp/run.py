import sys, os
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

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

from utils.ds import PairDataset
from utils.model import HourGlassMLP, ConventionalMLP
from utils.train_eval import train_and_eval

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp_home', type=str, default='./results')
    
    # experiment settings
    parser.add_argument('--ds_name', type=str, default='mnist', choices=['mnist', 'cifar100', 'bsds500', 'imagenet32'])
    parser.add_argument('--mode', type=str, required=True, choices=['generative_classification', 'denoising', 'super_resolution'])
    parser.add_argument('--noise_std', type=float, default=0.25)
    parser.add_argument('--down_scale', type=float)
    
    # MLP type
    parser.add_argument('--model_type', type=str, default='hourglass', choices=['hourglass', 'conventional'])
        
    # model architectural hyperparameters
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--hidden_dims', nargs='+', type=int, required=True, help='List of middle dims')
    ## W_in related variants
    parser.add_argument('--wo_Win', action='store_true', default=False, help='Disable input projection')
    parser.add_argument('--fix_Win', action='store_true', default=False, help='Set input projection to non-trinable')
    parser.add_argument('--I_Win', action='store_true', default=False, help='Set input projection to Identity')
    ## W_out related variants
    parser.add_argument('--wo_Wout', action='store_true', default=False, help='Disable output projection')
    
    # training hyperparmeters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                    help='Enable data augmentation')
    parser.add_argument('--aug_num', type=int, default=2, choices=[2, 4])

    # experimtal seed
    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    
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

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params

def main():
    #############
    # Basic setup
    #############
    args = parse_arguments()
    device = args.device
    print(device)
   
    if args.ds_name in ['mnist']:
        input_dim = 784
        output_dim = 784
        logging_step = 'epoch_based'
        args.down_scale = 2.0  
    elif args.ds_name in ['imagenet32']:
        input_dim = 3072
        output_dim = 3072
        logging_step = 'step_based'
        args.down_scale = 2.0
    else:
        raise NotImplementedError()
    
    if args.mode == 'super_resolution':
        input_dim = int(input_dim // (args.down_scale) ** 2)
        
    if args.model_type == 'hourglass':
        if args.wo_Win or args.wo_Wout:
            raise ValueError('Hourglass models require input and output projection to match the dimensions')
        if args.I_Win:
            raise NotImplementedError('It is possible to do this, but not implemented yet')
        model = HourGlassMLP(input_dim=input_dim, 
                                output_dim=output_dim,
                                latent_dim=args.latent_dim, 
                                hidden_dims=args.hidden_dims, 
                                fix_Win=args.fix_Win).to(device)
    elif args.model_type == 'conventional':
        model = ConventionalMLP(input_dim=input_dim, 
                                output_dim=output_dim,
                                latent_dim=args.latent_dim, 
                                hidden_dims=args.hidden_dims, 
                                wo_Win=args.wo_Win, 
                                fix_Win=args.fix_Win, 
                                I_Win=args.I_Win,
                                wo_Wout=args.wo_Wout).to(device)
    
    print(model)
    total_params, trainable_params = count_parameters(model)

    # Setup exp name
    hidden_dims_str = '_'.join(map(str, args.hidden_dims))
    model_type_str = args.model_type
    ## handle Win first
    if args.wo_Win:
        print('Training with no W_in matrix!')
        model_type_str += "_wo_Win"
    else:
        print('Training with W_in matrix!')
        model_type_str += "_w_Win"
        if args.fix_Win:
            model_type_str += "_fix"
        if args.I_Win:
            model_type_str += "_I"
    ## then Wout
    if args.wo_Wout:
        print('Training with no W_out matrix!')
        model_type_str += "_wo_Wout"

    if args.mode == 'denoising':
        if args.use_augmentation:
            exp_dir = f"{args.exp_home}/{args.ds_name}_{args.mode}_std{args.noise_std}/bs{args.batch_size}_ep{args.epochs}_aug{args.aug_num}/{model_type_str}_{trainable_params}/latent{args.latent_dim}_hidden{hidden_dims_str}/lr{args.lr}_run{args.run_id}"
        else:
            exp_dir = f"{args.exp_home}/{args.ds_name}_{args.mode}_std{args.noise_std}/bs{args.batch_size}_ep{args.epochs}/{model_type_str}_{trainable_params}/latent{args.latent_dim}_hidden{hidden_dims_str}/lr{args.lr}_run{args.run_id}"
    elif args.mode == 'super_resolution':
        if args.use_augmentation:
            exp_dir = f"{args.exp_home}/{args.ds_name}_{args.mode}_down{args.down_scale}/bs{args.batch_size}_ep{args.epochs}_aug{args.aug_num}/{model_type_str}_{trainable_params}/latent{args.latent_dim}_hidden{hidden_dims_str}/lr{args.lr}_run{args.run_id}"
        else:
            exp_dir = f"{args.exp_home}/{args.ds_name}_{args.mode}_down{args.down_scale}/bs{args.batch_size}_ep{args.epochs}/{model_type_str}_{trainable_params}/latent{args.latent_dim}_hidden{hidden_dims_str}/lr{args.lr}_run{args.run_id}"        
    elif args.mode == 'generative_classification':
        if args.use_augmentation:
            exp_dir = f"{args.exp_home}/{args.ds_name}_{args.mode}/bs{args.batch_size}_ep{args.epochs}_aug{args.aug_num}/{model_type_str}_{trainable_params}/latent{args.latent_dim}_hidden{hidden_dims_str}/lr{args.lr}_run{args.run_id}"
        else:
            exp_dir = f"{args.exp_home}/{args.ds_name}_{args.mode}/bs{args.batch_size}_ep{args.epochs}/{model_type_str}_{trainable_params}/latent{args.latent_dim}_hidden{hidden_dims_str}/lr{args.lr}_run{args.run_id}"
    
    # If the exp dir already exist, return error
    print(f'Experiment directory: {exp_dir}')
    if os.path.exists(os.path.join(exp_dir, 'experiment_summary.json')):
        raise ValueError(f"Dupicated experiment: [{exp_dir}] already exist and contains summary json file!")
    else:
        os.makedirs(exp_dir, exist_ok=True)
    
    
    ###############################
    # Load train, val, and test set
    ###############################
    train_set = PairDataset(args.ds_name, mode=args.mode, split='train', 
                            noise_std=args.noise_std, down_scale=args.down_scale,
                            use_augmentation=args.use_augmentation, aug_num=args.aug_num)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    eval_set = PairDataset(args.ds_name, mode=args.mode, split='eval', noise_std=args.noise_std, down_scale=args.down_scale, use_augmentation=False)
    eval_loader = DataLoader(eval_set, batch_size=128, shuffle=False)

    test_set = PairDataset(args.ds_name, mode=args.mode, split='test', noise_std=args.noise_std, down_scale=args.down_scale, use_augmentation=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    img_shape = (train_set.C, train_set.H, train_set.W)

    if args.mode == 'generative_classification':
        # set prototype images
        eval_loader.dataset.prototypes = train_set.prototypes
        test_loader.dataset.prototypes = train_set.prototypes

        # save prototype images
        gt_image_paths = {}
        gt_dir = os.path.join(exp_dir, "gt_images")
        os.makedirs(gt_dir, exist_ok=True)
        for label, img in train_set.prototypes.items():
            img_path = os.path.join(gt_dir, f"gt_{label}.png")
            ndarr = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
            if img.shape[0] == 1:
                Image.fromarray(ndarr.squeeze(), mode='L').save(img_path)
            else:
                Image.fromarray(ndarr, mode='RGB').save(img_path)
            gt_image_paths[label] = img_path

    ################
    # Model Training
    ################
    set_seed(args.seed + args.run_id) # set different seed for different run id        
    training_logs, test_results = train_and_eval(
                            model, args.model_type, trainable_params, total_params, 
                            train_loader, eval_loader, test_loader,
                            img_shape, exp_dir, args.mode, args.down_scale,
                            args.device,
                            args.epochs, args.lr, min_lr=0.0,
                            logging_step=logging_step,
                            eval_interval_epochs=50,
                            log_interval_steps=2500,
                            eval_interval_steps=5000)

    
    ################
    # Results Saving
    ################
    summary = {
        "args": vars(args),              
        "training_logs": training_logs,  
        "test_results": test_results,    
    }

    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, "experiment_summary.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved results to {save_path}")

if __name__ == '__main__':
    main()