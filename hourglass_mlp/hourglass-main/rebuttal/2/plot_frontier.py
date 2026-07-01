import argparse
import os
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np


# ---------------------- CLI ----------------------

def parse_arguments():
    p = argparse.ArgumentParser(description='Plot Pareto frontier from new JSON structure')
    p.add_argument('--json_path', type=str, required=True)
    p.add_argument('--plot_output', type=str, default='./rebuttal/2/plots')
    p.add_argument('--metric', type=str, default='test_psnr', choices=['test_psnr', 'eval_psnr'])
    p.add_argument('--bs_ep', type=str, default=None, help='Filter by bs_ep (e.g., bs128_ep30). If None, use all.')
    p.add_argument('--x_min', type=float, default=None)
    p.add_argument('--x_max', type=float, default=None)
    p.add_argument('--y_min', type=float, default=None)
    p.add_argument('--y_max', type=float, default=None)
    p.add_argument('--title', type=str, default=None)
    p.add_argument('--std_scale', type=float, default=1.0)
    p.add_argument('--show_arch', action='store_true', default=False, help='Show architecture info (latent_dim, hidden_dim, L) near each point')
    return p.parse_args()


# ---------------------- Helpers ----------------------

def extract_model_type_from_config(config_name):
    """
    Extract model type from config name by removing the trailing parameter count
    Examples:
      'conventional_wo_Win_wo_Wout_2304960' -> 'conventional_wo_Win_wo_Wout'
      'conventional_w_Win_I_3032512' -> 'conventional_w_Win_I'
    """
    match = re.match(r'(.+?)_(\d+)$', config_name)
    if match:
        return match.group(1)
    return config_name


def extract_params_from_config_name(config_name):
    """Extract parameter count from config name like 'conventional_wo_Win_2417856'"""
    match = re.search(r'(\d+)$', config_name)
    if match:
        return int(match.group(1))
    return None


def get_model_type_style(model_type):
    """
    Return color, marker, and label for each model type
    Add more mappings as needed
    """
    style_map = {
        'conventional_w_Win': {
            'color': '#D55E00',
            'marker': 'o',
            'label': r'Conventional (w/ $W_{in}, W_{out}$)'
        },
        'conventional_wo_Win_wo_Wout': {
            'color': '#B9CE2F',
            'marker': 'o',
            'label': r'Conventional (w/o $W_{in}, W_{out}$)',
            'linestyle': ':'
        },
        'conventional_w_Win_wo_Wout': {
            'color': "#A9AB1F",
            'marker': 'o',
            'label': r'Conventional (w/o $W_{out}$)',
            'linestyle': ':'
        },
        'hourglass': {
            'color': '#0072B2',
            'marker': 'D',
            'label': 'Hourglass'
        },
        # 'hourglass_w_Win_fix': {
        #     'color': "#11C3A2",
        #     'marker': 'D',
        #     'label': r'Hourglass (fix $W_{in}$)',
        #     'linestyle': ':'
        # }
    }
    if model_type in style_map:
        return style_map[model_type]

    else:
        return None  # 返回 None 而非默認樣式


def extract_grouped_stats(data, metric_name, bs_ep_filter=None):
    """
    Extract stats grouped by (bs_ep, model_type, latent_hidden, lr) -> aggregated over runs
    Returns: dict[model_type] -> list of dicts with {x, mean, std, vals, ...}
    
    For eval_psnr: takes the last element of the list
    For test_psnr: takes the single value
    bs_ep_filter: if provided, only include data from this bs_ep configuration
    """
    grouped = {}  # key: (bs_ep, model_type, latent_hidden, lr) -> list[values]
    
    for bs_ep_key, bs_ep_data in data.items():
        # Filter by bs_ep if specified
        if bs_ep_filter is not None and bs_ep_key != bs_ep_filter:
            continue
            
        for model_config, latent_hidden_data in bs_ep_data.items():
            raw_params = extract_params_from_config_name(model_config)
            if raw_params is None:
                continue
            
            # Extract model type (remove parameter count)
            model_type = extract_model_type_from_config(model_config)
            
            for latent_hidden, lr_data in latent_hidden_data.items():
                for lr_key, run_data in lr_data.items():
                    # Parse lr
                    lr_val = None
                    if lr_key.startswith('lr_'):
                        try:
                            lr_val = float(lr_key.split('lr_')[1])
                        except:
                            continue
                    
                    # Collect values from all runs
                    for run_key, run_info in run_data.items():
                        metrics = run_info.get('metrics', {})
                        if metric_name in metrics:
                            value = metrics[metric_name]
                            
                            # Handle list vs single value
                            if isinstance(value, list):
                                if len(value) > 0:
                                    y = value[-1]  # Take last element
                                else:
                                    continue
                            else:
                                y = value
                            
                            if y is not None and np.isfinite(y):
                                key = (bs_ep_key, model_type, latent_hidden, lr_val, raw_params)
                                grouped.setdefault(key, []).append(float(y))
    
    # Aggregate by model type
    stats_by_type = {}
    for (bs_ep, model_type, latent_hidden, lr_val, raw_params), vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        
        if model_type not in stats_by_type:
            stats_by_type[model_type] = []
            
        stats_by_type[model_type].append({
            'x': raw_params / 1e6,
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'vals': list(vals),
            'bs_ep': bs_ep,
            'model_type': model_type,
            'latent_hidden': latent_hidden,
            'lr': lr_val,
            'raw_params': raw_params
        })
    
    return stats_by_type


def best_per_x_from_stats(stats_list):
    """Select best config per x based on mean (higher is better for PSNR)"""
    if not stats_list:
        return []
    
    by_x = {}
    for r in stats_list:
        x = r['x']
        mean = r['mean']
        if not (np.isfinite(x) and np.isfinite(mean)):
            continue
        if x not in by_x:
            by_x[x] = r
        else:
            if mean > by_x[x]['mean']:  # Higher is better for PSNR
                by_x[x] = r
    
    out = list(by_x.values())
    out.sort(key=lambda t: t['x'])
    return out


def upper_envelope_stats(best_by_x, eps=1e-12):
    """Extract upper envelope (non-decreasing for PSNR)"""
    if not best_by_x:
        return []
    
    env = []
    best_y = -float('inf')
    for r in best_by_x:
        if r['mean'] > best_y + eps:
            env.append(r)
            best_y = r['mean']
    return env


def _build_poly_ribbon(turn_xs, turn_means, turn_stds):
    """Build ribbon boundaries for fill_between"""
    if not turn_xs:
        return np.array([]), np.array([]), np.array([])
    X = np.asarray(turn_xs, dtype=float)
    M = np.asarray(turn_means, dtype=float)
    S = np.asarray(turn_stds, dtype=float)
    return X, M - S, M + S


# ---------------------- Plot ----------------------
def extract_latent_hidden_L(latent_hidden_str):
    """
    從 latent_hidden 字串中提取 latent, hidden, 和 L (重複次數)
    例如: 
        - 'latent80_hidden120_120_120' -> latent=80, hidden=120, L=3
        - 'latent64_hidden96' -> latent=64, hidden=96, L=1
    """
    # 提取 latent 部分
    latent_match = re.search(r'latent(\d+)', latent_hidden_str)
    latent = int(latent_match.group(1)) if latent_match else None
    
    # 提取 hidden 部分
    hidden_match = re.search(r'hidden(.+)$', latent_hidden_str)
    if hidden_match:
        hidden_part = hidden_match.group(1)
        # 分割成數字列表
        hidden_values = hidden_part.split('_')
        
        if len(hidden_values) > 0:
            # 第一個值作為 hidden
            hidden = int(hidden_values[0])
            # L 是重複的次數
            L = len(hidden_values)
        else:
            hidden = None
            L = 0
    else:
        hidden = None
        L = 0
    
    return latent, hidden, L

def plot_pareto(metric_name, title, stats_by_type, output_path,
                x_min=None, x_max=None, y_min=None, y_max=None, std_scale=1.0, show_arch=False):
    """Plot Pareto frontier with polyline and std ribbon for each model type"""
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(14, 12))
    
    legend_handles = []
    y_all_means = []
    
    # 存储每种模型类型的数据点和架构信息，以便稍后添加标签
    model_data = {}
    
    # Plot each model type separately
    for model_type, stats in stats_by_type.items():
        if not stats:
            continue
        # Get style for this model type
        style = get_model_type_style(model_type)
        if style is None:  # 跳過未定義的 model type
            print(f"[Skip] Model type '{model_type}' not in style_map, skipping...")
            continue
        color = style['color']
        marker = style['marker']
        label = style['label']
        linestyle = style.get('linestyle', '-')  # 默認使用實線

        # Compute frontier
        best_list = best_per_x_from_stats(stats)
        if not best_list:
            continue
        
        frontier = upper_envelope_stats(best_list)
        if not frontier:
            continue
        
        # Extract X/M/S
        X = np.array([r['x'] for r in frontier], dtype=float)
        M = np.array([r['mean'] for r in frontier], dtype=float)
        S = np.array([r['std'] for r in frontier], dtype=float) * float(std_scale)
        
        # Filter by x range if specified
        if (x_min is not None) or (x_max is not None):
            lo = -np.inf if x_min is None else float(x_min)
            hi = np.inf if x_max is None else float(x_max)
            mask = (X >= lo) & (X <= hi)
            X, M, S = X[mask], M[mask], S[mask]
        
        if X.size == 0:
            continue
        
        # Plot std ribbon
        Xb, Lb, Ub = _build_poly_ribbon(list(X), list(M), list(S))
        ax = plt.gca()
        ax.fill_between(Xb, Lb, Ub, color=color, alpha=0.20, zorder=1,
                         linewidth=0, edgecolor='none', antialiased=True)
        
        # Plot polyline
        line, = plt.plot(X, M, color=color, alpha=1.0, linewidth=2.5, 
                linestyle=linestyle, zorder=5, label=label)
        
        # Plot scatter points
        plt.scatter(X, M, color=color, alpha=1.0, marker=marker, s=120,
                    linewidths=0.4, zorder=6)
        
        # 存储该模型类型的数据点和架构信息
        if show_arch:
            arch_info = []
            for i, r in enumerate(frontier):
                if i < len(X):
                    latent, hidden, L = extract_latent_hidden_L(r['latent_hidden'])
                    if latent is not None and hidden is not None:
                        arch_info.append((X[i], M[i], f"({latent}, {hidden}, {L})"))
            
            model_data[model_type] = {
                'color': color,
                'arch_info': arch_info
            }
        
        legend_handles.append(line)
        y_all_means.extend(list(M))
        
        # Print statistics
        print(f"\n[Model Type={model_type}] Metric={metric_name}")
        print("  Per-x grouped statistics (over runs):")
        for r in sorted(stats, key=lambda t: (t['x'], t['latent_hidden'])):
            runs_str = '[' + ', '.join(f'{v:.3f}' for v in r['vals']) + ']'
            print(f"    x={r['x']:.3f}M | {r['bs_ep']}/{r['latent_hidden']} | lr={r['lr']} -> mean={r['mean']:.3f}, std={r['std']:.3f}, runs={runs_str}")
        
        print("  Frontier:")
        print(f"    {'x (M)':<10} {'mean':<10} {'latent':<10} {'hidden':<10} {'L':<5} {'LR':<10}")
        print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*5} {'-'*10}")
        for r in frontier:
            latent, hidden, L = extract_latent_hidden_L(r['latent_hidden'])
            print(f"    {r['x']:<10.3f} {r['mean']:<10.3f} {latent:<10} {hidden:<10} {L:<5} {r['lr']:<10.5f} ")
 
    if not y_all_means:
        print(f"[Warn] No data to plot for metric={metric_name}")
        return
    
    # Labels and styling
    plt.xlabel('Parameter Counts (M)', fontsize=40, labelpad=20)
    ylabel = 'PSNR (dB)' if 'psnr' in metric_name.lower() else metric_name.replace('_', ' ').upper()
    plt.ylabel(ylabel, fontsize=40, labelpad=20)
    
    plt.grid(True, alpha=0.3, linewidth=1)
    plt.xticks(fontsize=38)
    plt.yticks(fontsize=38)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    
    # Set axis limits
    if (x_min is not None) or (x_max is not None):
        cur = ax.get_xlim()
        plt.xlim(x_min if x_min is not None else cur[0],
                 x_max if x_max is not None else cur[1])
    
    if (y_min is None) or (y_max is None):
        ymin = float(np.min(y_all_means))
        ymax = float(np.max(y_all_means))
        yr = ymax - ymin
        margin = (yr * 0.06) if yr != 0 else max(1.0, abs(ymin) * 0.05)
        plt.ylim(ymin - margin, ymax + margin)
    else:
        cur = ax.get_ylim()
        plt.ylim(y_min if y_min is not None else cur[0],
                 y_max if y_max is not None else cur[1])
    
    # 现在添加架构信息标签，使用自动计算的偏移量
    if show_arch:
        print("\nAdding architecture info labels to the plot...")
        # 获取当前的坐标轴范围
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        # 计算自适应偏移量（根据图的大小调整）
        x_offset = x_range * 0.01  # x轴范围的1%
        y_offset = y_range * 0.01  # y轴范围的1%
        
        # 为每个模型类型添加标签
        for model_type, data in model_data.items():
            for x, y, text in data['arch_info']:
                plt.text(x + x_offset, y - y_offset, text, 
                         fontsize=10, color=data['color'], weight='bold')
    
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    
    # Legend
    if legend_handles:
        leg = plt.legend(handles=legend_handles,
                         loc='lower right',
                        #  fontsize=28,
                         fontsize=26,
                         frameon=True)
        lh_list = getattr(leg, "legend_handles", None)
        if lh_list is None:
            lh_list = getattr(leg, "legendHandles", [])
        for lh in lh_list:
            if isinstance(lh, Line2D):
                lh.set_markeredgecolor('white')
                lh.set_markeredgewidth(0.8)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
    
    print(f"\nSaved Pareto frontier plot: {output_path}")


# ---------------------- Main ----------------------

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.json_path):
        print(f"JSON file does not exist: {args.json_path}")
        return
    
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON: {args.json_path}")
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return
    
    json_filename = Path(args.json_path).stem
    output_dir = Path(args.plot_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Extract stats grouped by model type
    stats_by_type = extract_grouped_stats(data, args.metric, args.bs_ep)
    
    if not stats_by_type:
        print(f"No data found for metric: {args.metric}" + 
              (f" with bs_ep: {args.bs_ep}" if args.bs_ep else ""))
        return
    
    total_configs = sum(len(stats) for stats in stats_by_type.values())
    print(f"Found {total_configs} grouped configurations for {args.metric}" +
          (f" (filtered by bs_ep: {args.bs_ep})" if args.bs_ep else ""))
    print(f"Model types found: {list(stats_by_type.keys())}")
    
    # Plot
    bs_ep_suffix = f"_{args.bs_ep}" if args.bs_ep else ""
    out_png = output_dir / f"{args.metric}_{json_filename}{bs_ep_suffix}.png"
    plot_pareto(args.metric, args.title, stats_by_type, out_png,
                x_min=args.x_min, x_max=args.x_max,
                y_min=args.y_min, y_max=args.y_max,
                std_scale=args.std_scale,
                show_arch=args.show_arch)
    
    print(f"\nPareto frontier plot generated!")
    print(f"Saved to: {output_dir}/")


if __name__ == '__main__':
    main()