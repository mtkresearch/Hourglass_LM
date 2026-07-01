import argparse
import os
import json
import re
from pathlib import Path
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_loc', type=str, default='./results', help='where the performance results at')
    parser.add_argument('--exp_folder', type=str, default='mnist_denoising_std0.25', help='the folder of the experiments')
    parser.add_argument('--json_output', type=str, default='./src/exp_record', help='the folder of the experiments')
    parser.add_argument('--sum_file', type=int, default=1)
    return parser.parse_args()

def extract_bs_ep_from_folder_name(folder_name):
    """
    Extract batch_size and epochs from folder name like 'bs128_ep30'
    """
    bs = None
    ep = None
    
    # Extract batch_size
    bs_match = re.search(r'bs(\d+)', folder_name)
    if bs_match:
        bs = int(bs_match.group(1))
    
    # Extract epochs
    ep_match = re.search(r'ep(\d+)', folder_name)
    if ep_match:
        ep = int(ep_match.group(1))
    
    return bs, ep

def extract_lr_and_run_from_folder_name(folder_name):
    """
    Extract learning rate and run number from folder name like 'lr0.001_run1'
    """
    lr = None
    run_num = 1  # default
    
    # Extract learning_rate (supports scientific notation)
    lr_match = re.search(r'lr([\d.]+(?:e[+-]?\d+)?)', folder_name)
    if lr_match:
        lr = float(lr_match.group(1))
    
    # Extract run number
    run_match = re.search(r'run(\d+)', folder_name)
    if run_match:
        run_num = int(run_match.group(1))
    
    return lr, run_num

def get_metrics_from_summary(experiment_summary_path):
    """
    Extract training_logs.psnr and test_results.test_psnr from experiment_summary.json
    """
    try:
        with open(experiment_summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = {}
        
        # Extract training_logs.psnr (list of values per epoch)
        if 'training_logs' in data and 'psnr' in data['training_logs']:
            metrics['eval_psnr'] = data['training_logs']['psnr']
        else:
            print(f"      Warning: training_logs.psnr not found")
            return None
        
        # Extract test_results.test_psnr (single value)
        if 'test_results' in data and 'test_psnr' in data['test_results']:
            metrics['test_psnr'] = data['test_results']['test_psnr']
        else:
            print(f"      Warning: test_results.test_psnr not found")
            return None
        
        return metrics
        
    except Exception as e:
        print(f"Error reading experiment_summary.json: {e}")
        return None

def main():
    args = parse_arguments()
    
    # Build full path
    experiment_path = os.path.join(args.exp_loc, args.exp_folder)
    print(f"Experiment path: {experiment_path}")
    
    if not os.path.exists(experiment_path):
        print(f"Experiment path does not exist: {experiment_path}")
        return
    
    experiment_path = Path(experiment_path)
    
    # Store organized data
    organized_data = {}
    
    print(f"Processing path: {experiment_path}")
    
    # Level 1: bs_ep folders like bs128_ep30
    for bs_ep_folder in experiment_path.iterdir():
        if not bs_ep_folder.is_dir():
            continue
        
        bs, ep = extract_bs_ep_from_folder_name(bs_ep_folder.name)
        
        if bs is None or ep is None:
            print(f"Skip unparseable bs_ep folder: {bs_ep_folder.name}")
            continue
        
        bs_ep_key = bs_ep_folder.name
        print(f"Processing bs_ep: {bs_ep_key}")
        
        organized_data[bs_ep_key] = {}
        
        # Group experiments by model configuration
        model_config_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Level 2: model config folders like conventional_wo_Win_2417856
        for model_config_folder in bs_ep_folder.iterdir():
            if not model_config_folder.is_dir():
                continue
            
            model_config_name = model_config_folder.name
            print(f"  Processing model config: {model_config_name}")
            
            # Level 3: latent/hidden config folders like latent784_hidden1150
            for latent_hidden_folder in model_config_folder.iterdir():
                if not latent_hidden_folder.is_dir():
                    continue
                
                latent_hidden_name = latent_hidden_folder.name
                print(f"    Processing latent_hidden: {latent_hidden_name}")
                
                # Level 4: lr_run folders like lr0.001_run1
                for lr_run_folder in latent_hidden_folder.iterdir():
                    if not lr_run_folder.is_dir():
                        continue
                    
                    # Extract learning rate and run number
                    lr, run_num = extract_lr_and_run_from_folder_name(lr_run_folder.name)
                    
                    if lr is None:
                        print(f"      Skip folder with unparseable lr: {lr_run_folder.name}")
                        continue
                    
                    # Check file existence
                    if args.sum_file == 1:
                        summary_file = lr_run_folder / 'experiment_summary.json'
                    else:
                        summary_file = lr_run_folder / f'experiment_summary_eval{args.sum_file}.json'
                    
                    if not summary_file.exists():
                        print(f"      experiment_summary.json not found: {lr_run_folder.name}")
                        continue
                    
                    # Read metrics from summary file
                    metrics = get_metrics_from_summary(summary_file)
                    
                    if metrics is None:
                        print(f"      Failed to read metrics: {lr_run_folder.name}")
                        continue
                    
                    # Add to corresponding group
                    model_config_groups[model_config_name][latent_hidden_name][lr].append({
                        'run': run_num,
                        'metrics': metrics
                    })
                    
                    print(f"      Successfully processed: {lr_run_folder.name}")
                    print(f"        Found: eval_psnr (epochs={len(metrics['eval_psnr'])}), test_psnr={metrics['test_psnr']:.4f}")
        
        # Organize grouped data
        for model_config_name, latent_hidden_groups in model_config_groups.items():
            organized_data[bs_ep_key][model_config_name] = {}
            
            for latent_hidden_name, lr_groups in latent_hidden_groups.items():
                organized_data[bs_ep_key][model_config_name][latent_hidden_name] = {}
                
                for lr, runs in lr_groups.items():
                    lr_key = f"lr_{lr}"
                    organized_data[bs_ep_key][model_config_name][latent_hidden_name][lr_key] = {}
                    
                    for run_data in runs:
                        run_key = f"run_{run_data['run']}"
                        organized_data[bs_ep_key][model_config_name][latent_hidden_name][lr_key][run_key] = {
                            'metrics': run_data['metrics']
                        }
    
    organized_data = dict(sorted(organized_data.items()))
    
    # Save JSON file
    output_dir = Path(args.json_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_filename = f"{args.exp_folder.replace('/', '_')}.json"
    output_file = output_dir / json_filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(organized_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Output statistics
    total_runs = 0
    total_configs = 0
    for bs_ep_key, bs_ep_data in organized_data.items():
        for model_config_name, model_config_data in bs_ep_data.items():
            for latent_hidden_name, latent_hidden_data in model_config_data.items():
                total_configs += 1
                for lr_key, lr_data in latent_hidden_data.items():
                    total_runs += len(lr_data)
    
    print(f"Statistics: Processed {total_configs} configurations, {total_runs} experiment runs")

if __name__ == '__main__':
    main()