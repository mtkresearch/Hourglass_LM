import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


def parse_arguments():
    p = argparse.ArgumentParser(description='Gather experiment results into summary JSON')
    p.add_argument('--results_dir', type=str, required=True, 
                   help='Path to results directory (e.g., results/mnist_generative_classification)')
    p.add_argument('--output_path', type=str, default='./summary.json',
                   help='Output path for summary JSON')
    p.add_argument('--metric', type=str, default='test_psnr',
                   help='Metric to extract (e.g., test_psnr, best_train_loss, best_eval_loss)')
    return p.parse_args()


def get_setup_key(dirname):
    """
    Return the full directory name as setup key.
    Examples: 'bs512_ep2_aug4', 'bs128_ep100', etc.
    """
    return dirname


def parse_model_type(dirname):
    """
    Parse model type from directory name starting with 'conventional_'.
    Extract the suffix after 'conventional_' and before any numeric suffix.
    
    Examples:
        'conventional_w_Win_12048512' -> 'w_Win'
        'conventional_wo_Win_12048512' -> 'wo_Win'
        'conventional_w_Win_train_random_12048512' -> 'w_Win_train_random'
        'conventional_hourglass_12048512' -> 'hourglass'
    
    Returns: config_type (str) or None
    """
    # Check if directory starts with 'conventional_'
    if not dirname.startswith('conventional_'):
        return None
    
    # Remove 'conventional_' prefix
    suffix = dirname[len('conventional_'):]
    
    # Remove numeric suffix (e.g., '_12048512')
    # Find the last part that's purely numeric
    parts = suffix.split('_')
    
    # Remove trailing numeric parts
    config_parts = []
    for part in parts:
        if part.isdigit() and len(part) > 4:  # Likely a parameter count
            break
        config_parts.append(part)
    
    if not config_parts:
        return None
    
    config_type = '_'.join(config_parts)
    return config_type


def parse_architecture(dirname):
    """
    Parse latent_dim, hidden_dim, and L from directory name.
    Example: 'latent784_hidden1150_1150_1150_1150_1150_1150'
    Returns: (latent_dim, hidden_dim, L)
    """
    # Extract latent dimension
    latent_match = re.search(r'latent(\d+)', dirname, re.IGNORECASE)
    latent_dim = int(latent_match.group(1)) if latent_match else None
    
    # Extract hidden dimensions
    hidden_match = re.search(r'hidden([\d_]+)', dirname, re.IGNORECASE)
    if hidden_match:
        hidden_str = hidden_match.group(1)
        hidden_dims = [int(x) for x in hidden_str.split('_') if x.isdigit()]
        if hidden_dims:
            # Count the number of layers (L)
            L = len(hidden_dims)
            # Assume all hidden dims are the same (take the first one)
            hidden_dim = hidden_dims[0]
            return latent_dim, hidden_dim, L
    
    return latent_dim, None, None


def parse_lr(dirname):
    """Extract learning rate from directory name like 'lr0.0001' or 'lr1e-4'"""
    # Try scientific notation first
    match = re.search(r'lr(\d+\.?\d*e[+-]?\d+)', dirname, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Try decimal notation
    match = re.search(r'lr(\d+\.?\d*)', dirname, re.IGNORECASE)
    if match:
        lr_val = float(match.group(1))
        # Convert to scientific notation format for consistency
        if lr_val == 0.0005:
            return 5e-4
        elif lr_val == 0.001:
            return 1e-3
        elif lr_val == 0.0001:
            return 1e-4
        return lr_val
    
    return None


def parse_run_number(dirname):
    """Extract run number from directory name like 'run1', 'run2', etc."""
    match = re.search(r'run(\d+)', dirname, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def format_lr_key(lr_value):
    """Format LR value as string key like 'lr_5e-4'"""
    if lr_value is None:
        return 'lr_unknown'
    
    # Format in scientific notation
    if lr_value >= 0.001:
        return f'lr_{lr_value:.0e}'.replace('e-0', 'e-')
    else:
        # For small values, use scientific notation
        exp = int(f'{lr_value:.0e}'.split('e')[1])
        mantissa = lr_value / (10 ** exp)
        if mantissa == 1.0:
            return f'lr_1e{exp}'
        else:
            return f'lr_{mantissa:.0f}e{exp}'


def extract_metric_value(summary, metric):
    """
    Extract the specified metric from experiment summary.
    
    Args:
        summary: dict containing experiment results
        metric: str, name of metric to extract
        
    Returns:
        float or None
    """
    # Handle special metrics from training_logs
    if metric == 'best_train_loss':
        train_losses = summary.get('training_logs', {}).get('train_losses', [])
        if train_losses:
            return min(train_losses)
        return None
    
    elif metric == 'best_eval_loss':
        eval_losses = summary.get('training_logs', {}).get('eval_losses', [])
        if eval_losses:
            return min(eval_losses)
        return None
    
    # Handle metrics from test_results (default behavior)
    else:
        test_results = summary.get('test_results', {})
        return test_results.get(metric)


def gather_results(results_dir, metric='test_psnr'):
    """
    Traverse the results directory and gather all experiment results.
    
    Structure:
    results/
      mnist_generative_classification/
        bs128_ep100/                          <- setup level (use full name as key)
          conventional_w_Win_train_random_12048512/  <- model type
            latent784_hidden1150_1150_1150_1150_1150_1150/  <- architecture
              lr0.0001_run1/                  <- lr and run
                experiment_summary.json
    
    Returns:
        Nested dict structure matching the target JSON format
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return {}
    
    # Structure: [setup_key][L][lr_key][config_type] = [list of values]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # Store dimension info per setup
    dimensions = {}
    
    # Traverse directory structure
    for setup_dir in results_dir.iterdir():
        if not setup_dir.is_dir():
            continue
        
        setup_key = get_setup_key(setup_dir.name)
        print(f"\nProcessing setup: {setup_key}")
        
        for model_dir in setup_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            config_type = parse_model_type(model_dir.name)
            if config_type is None:
                print(f"  Skipping (not conventional_*): {model_dir.name}")
                continue
            
            print(f"  Model type: {config_type} ({model_dir.name})")
            
            for arch_dir in model_dir.iterdir():
                if not arch_dir.is_dir():
                    continue
                
                latent_dim, hidden_dim, L = parse_architecture(arch_dir.name)
                if L is None:
                    print(f"    Skipping unknown architecture: {arch_dir.name}")
                    continue
                
                # Store dimension info
                if setup_key not in dimensions:
                    dimensions[setup_key] = {
                        'latent_dim': latent_dim,
                        'hidden_dim': hidden_dim
                    }
                
                print(f"    Architecture: latent={latent_dim}, hidden={hidden_dim}, L={L}")
                
                for run_dir in arch_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    
                    lr_value = parse_lr(run_dir.name)
                    run_num = parse_run_number(run_dir.name)
                    
                    if lr_value is None:
                        print(f"      Skipping unknown LR: {run_dir.name}")
                        continue
                    
                    lr_key = format_lr_key(lr_value)
                    
                    # Load experiment_summary.json
                    summary_path = run_dir / 'experiment_summary.json'
                    if not summary_path.exists():
                        print(f"      Missing summary: {summary_path}")
                        continue
                    
                    try:
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                        
                        # Extract metric value using the new function
                        metric_value = extract_metric_value(summary, metric)
                        
                        if metric_value is not None:
                            data[setup_key][str(L)][lr_key][config_type].append(float(metric_value))
                            print(f"      ✓ Run {run_num}, LR={lr_key}: {metric}={metric_value:.4f}")
                        else:
                            print(f"      ✗ Metric '{metric}' not found in {summary_path}")
                    
                    except Exception as e:
                        print(f"      Error reading {summary_path}: {e}")
    
    # Convert to final JSON structure
    output = {}
    for setup_key in sorted(data.keys()):
        output[setup_key] = {
            'dimension': dimensions.get(setup_key, {'latent_dim': None, 'hidden_dim': None}),
            'vary_L': {}
        }
        
        for L in sorted(data[setup_key].keys(), key=int):
            output[setup_key]['vary_L'][L] = {}
            
            for lr_key in sorted(data[setup_key][L].keys()):
                output[setup_key]['vary_L'][L][lr_key] = {}
                
                for config_type in sorted(data[setup_key][L][lr_key].keys()):
                    values = data[setup_key][L][lr_key][config_type]
                    output[setup_key]['vary_L'][L][lr_key][config_type] = values
    
    return output


def main():
    args = parse_arguments()
    
    print(f"Gathering results from: {args.results_dir}")
    print(f"Extracting metric: {args.metric}")
    
    # Gather all results
    summary = gather_results(args.results_dir, metric=args.metric)
    
    if not summary:
        print("\nNo results found!")
        return
    
    # Add metric suffix to output path
    output_path = Path(args.output_path)
    # Split filename and extension
    stem = output_path.stem  # e.g., 'summary'
    suffix = output_path.suffix  # e.g., '.json'
    parent = output_path.parent  # e.g., '.'
    
    # Create new filename with metric suffix
    new_filename = f"{stem}_{args.metric}{suffix}"
    output_path = parent / new_filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Summary saved to: {output_path}")
    
    # Print statistics
    print("\n=== Summary Statistics ===")
    for setup_key in summary.keys():
        print(f"\n{setup_key}:")
        print(f"  Dimensions: {summary[setup_key]['dimension']}")
        print(f"  L values: {list(summary[setup_key]['vary_L'].keys())}")
        
        for L, lr_data in summary[setup_key]['vary_L'].items():
            print(f"  L={L}:")
            for lr_key, config_data in lr_data.items():
                print(f"    {lr_key}:")
                for config_type, values in config_data.items():
                    print(f"      {config_type}: {len(values)} runs, mean={sum(values)/len(values):.4f}")


if __name__ == '__main__':
    main()