import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_arguments():
    p = argparse.ArgumentParser(description='Plot L vs metric with different W_in configurations')
    p.add_argument('--json_path', type=str, required=True, help='Path to JSON file')
    p.add_argument('--plot_output', type=str, default='./plots', help='Output directory for plots')
    p.add_argument('--setup', type=str, default='epoch_100',
                   help='Which epoch to plot')
    p.add_argument('--y_min', type=float, default=None, help='Y-axis minimum')
    p.add_argument('--y_max', type=float, default=None, help='Y-axis maximum')
    p.add_argument('--title', type=str, default=None, help='Plot title')
    p.add_argument('--metric', type=str, default=None, 
                   help='Metric type (auto-detected from filename if not specified)')
    return p.parse_args()


def detect_metric_from_filename(json_path):
    """
    Detect metric type from JSON filename.
    Examples:
        'summary_test_psnr.json' -> 'test_psnr'
        'summary_best_train_loss.json' -> 'best_train_loss'
        'summary_best_eval_loss.json' -> 'best_eval_loss'
    """
    filename = Path(json_path).stem  # Remove .json extension
    
    # Check for known metric patterns
    if 'best_train_loss' in filename:
        return 'best_train_loss'
    elif 'best_eval_loss' in filename:
        return 'best_eval_loss'
    elif 'test_psnr' in filename:
        return 'test_psnr'
    elif 'psnr' in filename.lower():
        return 'test_psnr'
    elif 'loss' in filename.lower():
        return 'best_train_loss'  # Default to loss type
    
    # Default to PSNR if cannot detect
    return 'test_psnr'


def is_higher_better(metric):
    """
    Determine if higher values are better for the given metric.
    Returns: True if higher is better, False if lower is better
    """
    # Metrics where higher is better
    higher_better_metrics = ['psnr', 'accuracy', 'precision', 'recall', 'f1']
    
    # Metrics where lower is better
    lower_better_metrics = ['loss', 'error', 'mse', 'mae']
    
    metric_lower = metric.lower()
    
    # Check if metric contains any higher-is-better keywords
    if any(keyword in metric_lower for keyword in higher_better_metrics):
        return True
    
    # Check if metric contains any lower-is-better keywords
    if any(keyword in metric_lower for keyword in lower_better_metrics):
        return False
    
    # Default: assume higher is better
    return True


def extract_data_from_json(data, epoch='epoch_100', metric='test_psnr'):
    """
    Extract data from JSON and organize by configuration type.
    For each L and configuration, select the LR with best mean value.
    
    Args:
        data: JSON data
        epoch: Which epoch configuration to use
        metric: Metric name (used to determine if higher or lower is better)
    
    Returns:
        dict: {config_name: {L: {'mean': float, 'std': float, 'vals': list}}}
    """
    if epoch not in data:
        print(f"Warning: {epoch} not found in data")
        return {}
    
    epoch_data = data[epoch]
    vary_L = epoch_data.get('vary_L', {})
    
    # Configuration types to plot
    config_types = [
        'w_Win',
        'w_Win_fix', 
        'w_Win_I', 
        'w_Win_fix_I',
        'wo_Win'
    ]
    
    results = {config: {} for config in config_types}
    
    # Determine if higher is better based on metric
    higher_is_better = is_higher_better(metric)
    print(f"\nMetric: {metric} ({'higher is better' if higher_is_better else 'lower is better'})")
    
    # Process each L value
    for L_str, L_data in vary_L.items():
        L = int(L_str)
        
        # For each configuration type
        for config in config_types:
            if higher_is_better:
                best_mean = -np.inf
                comparison = lambda new, best: new > best
            else:
                best_mean = np.inf
                comparison = lambda new, best: new < best
            
            best_lr_data = None
            best_lr_key = None
            
            # Check all LRs and find the one with best mean
            for lr_key, lr_data in L_data.items():
                if not isinstance(lr_data, dict):
                    continue
                
                if config in lr_data:
                    vals = lr_data[config]
                    if vals and len(vals) > 0:
                        mean_val = np.mean(vals)
                        if comparison(mean_val, best_mean):
                            best_mean = mean_val
                            best_lr_data = vals
                            best_lr_key = lr_key
            
            # Store the best LR's data for this L and config
            if best_lr_data is not None:
                results[config][L] = {
                    'mean': np.mean(best_lr_data),
                    'std': np.std(best_lr_data),
                    'vals': best_lr_data,
                    'best_lr': best_lr_key
                }
    
    return results


def get_y_label(metric):
    """Get appropriate Y-axis label based on metric type."""
    metric_lower = metric.lower()
    
    # Handle specific metric patterns
    if metric == 'best_train_loss':
        return 'Train Loss (Best)'
    elif metric == 'best_eval_loss':
        return 'Eval Loss (Best)'
    elif metric == 'test_psnr':
        return 'Test PSNR (dB)'
    elif 'train_loss' in metric_lower:
        return 'Train Loss'
    elif 'eval_loss' in metric_lower or 'val_loss' in metric_lower:
        return 'Eval Loss'
    elif 'test_loss' in metric_lower:
        return 'Test Loss'
    elif 'psnr' in metric_lower:
        return 'PSNR (dB)'
    elif 'accuracy' in metric_lower:
        return 'Accuracy'
    else:
        # Fallback: convert snake_case to Title Case
        return metric.replace('_', ' ').title()


def plot_L_vs_metric(results, output_path, metric='test_psnr', y_min=None, y_max=None, title=None):
    """
    Plot L vs metric with shadow regions for std.
    """
    # Set style parameters similar to the reference code
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(14, 12))
    
    # Define colors and labels for each configuration
    config_styles = {
        'w_Win': {
            'color': '#D55E00', 
            'label': 'w/ $W_{in}$ (Init~$N$, Train)',
            'marker': 'o'
        },
        'w_Win_fix': {
            'color': "#78A1B8", 
            'label': 'w/ $W_{in}$ (Init~$N$, Fix)',
            'marker': 's'
        },
        'w_Win_I': {
            'color': '#009E73', 
            'label': 'w/ $W_{in}$ (Init=$I$, Train)',
            'marker': '^'
        },
        'w_Win_fix_I': {
            'color': '#CC79A7', 
            'label': 'w/ $W_{in}$ (Init=$I$, Fix)',
            'marker': 'D'
        },
        'wo_Win': {
            'color': '#999999', 
            'label': 'w/o $W_{in}$',
            'marker': 'v'
        }
    }
    
    ax = plt.gca()
    
    # Plot each configuration
    for config, style in config_styles.items():
        if config not in results or not results[config]:
            continue
        
        # Sort by L value
        L_values = sorted(results[config].keys())
        means = [results[config][L]['mean'] for L in L_values]
        stds = [results[config][L]['std'] for L in L_values]
        
        L_array = np.array(L_values)
        mean_array = np.array(means)
        std_array = np.array(stds)
        
        # Plot line with markers
        line, = plt.plot(L_array, mean_array, 
                        color=style['color'],
                        marker=style['marker'],
                        markersize=10,
                        linewidth=2.5,
                        label=style['label'],
                        alpha=1.0,
                        zorder=5)
        
        # Plot shadow region for std
        ax.fill_between(L_array, 
                       mean_array - std_array, 
                       mean_array + std_array,
                       color=style['color'],
                       alpha=0.20,
                       linewidth=0,
                       zorder=1)
        
        # Print statistics
        print(f"\n[{config}]")
        for L in L_values:
            data = results[config][L]
            vals_str = '[' + ', '.join(f'{v:.4f}' for v in data['vals']) + ']'
            best_lr = data.get('best_lr', 'unknown')
            print(f"  L={L} (best_lr={best_lr}): mean={data['mean']:.4f}, std={data['std']:.4f}, vals={vals_str}")
    
    # Set labels and title
    plt.xlabel('Number of Layers (L)', fontsize=40, labelpad=20)
    plt.ylabel(get_y_label(metric), fontsize=40, labelpad=20)
    
    if title:
        plt.title(title, fontsize=32, pad=25)
    
    # Collect all unique L values from all configurations
    all_L_values = set()
    for config in config_styles.keys():
        if config in results and results[config]:
            all_L_values.update(results[config].keys())

    all_L_values = sorted(all_L_values)

    # Grid and ticks
    plt.grid(True, alpha=0.3, linewidth=1)
    plt.xticks(all_L_values, fontsize=38)  # 明確設定 x 軸刻度
    plt.yticks(fontsize=38)

    # Set x-axis to show integer values
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    # Set axis limits
    if y_min is not None or y_max is not None:
        cur_ylim = ax.get_ylim()
        plt.ylim(y_min if y_min is not None else cur_ylim[0],
                y_max if y_max is not None else cur_ylim[1])
    
    # Set spine width
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    
    # Add legend
    leg = plt.legend(loc='best',
                    #  loc='lower right',
                    fontsize=20,
                    frameon=True)
    
    # Tight layout and save
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Reset font size
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
    
    print(f"\n已保存圖表: {output_path}")


def main():
    args = parse_arguments()
    
    # Check if JSON file exists
    if not os.path.exists(args.json_path):
        print(f"JSON檔案不存在: {args.json_path}")
        return
    
    # Detect or use specified metric
    if args.metric:
        metric = args.metric
        print(f"使用指定的 metric: {metric}")
    else:
        metric = detect_metric_from_filename(args.json_path)
        print(f"從檔名自動偵測 metric: {metric}")
    
    # Load JSON data
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功讀取JSON檔案: {args.json_path}")
    except Exception as e:
        print(f"讀取JSON檔案時發生錯誤: {e}")
        return
    
    # Extract data
    results = extract_data_from_json(data, epoch=args.setup, metric=metric)
    
    if not any(results.values()):
        print(f"No data found for {args.setup}")
        return
    
    # Create output directory
    output_dir = Path(args.plot_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    json_filename = Path(args.json_path).stem
    output_path = output_dir / f"{json_filename}_{args.setup}_L_vs_metric.png"
    
    # Plot
    plot_L_vs_metric(results, output_path, metric=metric,
                     y_min=args.y_min, y_max=args.y_max, 
                     title=args.title)
    
    print(f"\n圖表已生成完成！")
    print(f"保存位置: {output_path}")


if __name__ == '__main__':
    main()