import json
import argparse
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional, Tuple, List, DefaultDict
from collections import defaultdict
import re
import numpy as np

# -------------------------- Config --------------------------
JSON_NAME = "experiment_summary.json"  # 你給的 eval_2 命名
_LR_KEYS = {"lr", "learning_rate", "init_lr", "base_lr", "initial_lr", "peak_lr"}

# LOSS_STEP = 500      # train_loss 的步長間隔
# METRIC_STEP = 2500   # eval_loss, psnr, ssim 的步長間隔
# LOSS_STEP = 50      # train_loss 的步長間隔
# METRIC_STEP = 500   # eval_loss, psnr, ssim 的步長間隔

# LOSS_STEP = 50      # train_loss 的步長間隔
# METRIC_STEP = 50   # eval_loss, psnr, ssim 的步長間隔

LOSS_STEP = 500      # train_loss 的步長間隔
METRIC_STEP = 2500   # eval_loss, psnr, ssim 的步長間隔

# -------------------------- Utils --------------------------

def _to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None

def find_lr_in_json(obj):
    """DFS 在 JSON 任何位置找常見 lr key，回傳第一個可轉成 float 的值"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in _LR_KEYS:
                val = _to_float(v)
                if val is not None:
                    return val
            found = find_lr_in_json(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = find_lr_in_json(it)
            if found is not None:
                return found
    return None

def parse_lr_from_path(path: str):
    """從路徑名稱猜 lr，例如 .../lr=3e-4/, .../lr0.001/, .../LR-0.0007_run2"""
    m = re.search(r'lr(?:\s*[:=_-]\s*|)([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)', path, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def fmt_lr(lr: float | None) -> str:
    if lr is None:
        return "?"
    return f"{lr:.1e}" if lr < 1e-2 else f"{lr:g}"

def safe_read_json(json_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 無法讀取 {json_path}: {e}")
        return None

def list_all_json_under(root: str) -> List[str]:
    hits = []
    if os.path.isfile(root) and os.path.basename(root) == JSON_NAME:
        hits.append(root)
        return hits
    for dirpath, _, filenames in os.walk(root):
        if JSON_NAME in filenames:
            hits.append(os.path.join(dirpath, JSON_NAME))
    return hits

# -------------------------- PSNR Extraction --------------------------

def extract_final_test_psnr(data: Dict[str, Any]) -> Optional[float]:
    # 新增：支持 test_results.test_psnr
    tr = data.get("test_results")
    if isinstance(tr, dict):
        # 優先檢查 test_psnr
        if isinstance(tr.get("test_psnr"), (int, float)):
            return float(tr["test_psnr"])
        
        # 原有的檢查
        ov = tr.get("overall")
        if isinstance(ov, dict):
            for k in ["psnr_batch", "psnr", "testing_psnr", "psnr_test", "test_psnr"]:
                v = ov.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
        for k in ["psnr_batch", "psnr", "testing_psnr", "psnr_test"]:
            v = tr.get(k)
            if isinstance(v, (int, float)):
                return float(v)

    # 舊相容
    tm = data.get("test_metrics")
    if isinstance(tm, dict):
        for k in ["psnr", "psnr_batch", "testing_psnr", "psnr_test", "test_psnr"]:
            v = tm.get(k)
            if isinstance(v, (int, float)):
                return float(v)

    best = data.get("best")
    if isinstance(best, dict):
        bt = best.get("test") if isinstance(best.get("test"), dict) else best
        if isinstance(bt, dict):
            for k in ["psnr", "psnr_batch", "testing_psnr", "psnr_test", "test_psnr"]:
                v = bt.get(k)
                if isinstance(v, (int, float)):
                    return float(v)

    om = data.get("overall_metrics")
    if isinstance(om, dict):
        for k in ["test_psnr", "psnr_test", "testing_psnr", "psnr_batch", "psnr"]:
            v = om.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        for k in ["psnr_batch", "psnr", "psnr_val", "val_psnr", "psnr_valid"]:
            if isinstance(om.get(k), list) and len(om[k]) > 0:
                return float(np.nanmax(np.array(om[k], dtype=float)))
    return None

# 保留但繪圖流程會改用聚合後的最佳 lr
def pick_best_experiment(root_or_run_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[float]]:
    candidates = list_all_json_under(root_or_run_path)
    if not candidates:
        print(f"[ERROR] 找不到 {JSON_NAME} in {root_or_run_path}")
        return None, None, None

    best_tuple = (None, None, None)   # (json_path, data, psnr)
    best_lr: Optional[float] = None

    for jp in candidates:
        data = safe_read_json(jp)
        if data is None:
            continue

        psnr = extract_final_test_psnr(data)
        if psnr is None:
            continue

        lr_val = find_lr_in_json(data)
        if lr_val is None:
            lr_val = parse_lr_from_path(jp)

        if best_tuple[2] is None or psnr > best_tuple[2]:
            best_tuple = (jp, data, psnr)
            best_lr = lr_val

    if best_tuple[0]:
        print(f"[INFO] (單 run) 最佳實驗: {best_tuple[0]} | testing PSNR = {best_tuple[2]:.4f} | lr = {fmt_lr(best_lr)}")

    return best_tuple

# -------------------------- Series Helpers --------------------------

def get_overall_metrics(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """修改：優先從 training_logs 提取，否則用 overall_metrics"""
    # 新結構：training_logs
    tl = data.get("training_logs")
    if isinstance(tl, dict):
        return tl
    
    # 舊結構：overall_metrics (向後兼容)
    om = data.get("overall_metrics")
    return om if isinstance(om, dict) else {}

def fetch_series(metrics: Dict[str, Any], possible_keys: List[str]) -> Optional[List[float]]:
    for k in possible_keys:
        v = metrics.get(k)
        if isinstance(v, list) and len(v) > 0:
            arr = np.array(v, dtype=float).tolist()  # 轉 float + 容忍 NaN
            return arr
    return None

def pad_and_stack(list_of_lists: List[List[float]]) -> np.ndarray:
    """將不同長度的序列用 NaN 補齊，回傳 shape=(num_runs, max_len) 的陣列"""
    if not list_of_lists:
        return np.empty((0, 0))
    max_len = max(len(x) for x in list_of_lists)
    out = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, seq in enumerate(list_of_lists):
        n = len(seq)
        out[i, :n] = np.array(seq, dtype=float)
    return out

def nanmean_std(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """對 axis=0 計算 nanmean / nanstd"""
    if arr.size == 0:
        return np.array([]), np.array([])
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    return mean, std

def steps_k(num_points: int, step_size: int) -> np.ndarray:
    """產生以 step_size 為間隔的步數（從 step_size 開始），並轉成 k 單位（除以 1000）"""
    if num_points <= 0:
        return np.array([])
    steps = np.arange(1, num_points + 1, dtype=float) * step_size
    return steps / 1000.0  # k 單位

# --------- Outlier filtering (per-run, per-series) ---------

def _first_finite(seq: List[float]) -> Optional[float]:
    for v in seq:
        vf = float(v)
        if not np.isnan(vf):
            return vf
    return None

def _apply_outlier_mask(seq: List[float], is_loss: bool) -> List[float]:
    """
    不做任何过滤，直接返回原始序列
    """
    if not seq:
        return []
    arr = np.array(seq, dtype=float)
    return arr.tolist()

# -------------------------- Group & Aggregate --------------------------

def group_runs_by_lr(root: str):
    """
    回傳:
      groups: dict[lr] -> { "train_loss": [list per run], "eval_loss": [...], "psnr": [...], "ssim": [...] }
      test_psnr_map: dict[lr] -> [float per run]
    """
    json_files = list_all_json_under(root)
    groups: DefaultDict[float, Dict[str, List[List[float]]]] = defaultdict(lambda: defaultdict(list))
    test_psnr_map: DefaultDict[float, List[float]] = defaultdict(list)

    for jp in json_files:
        data = safe_read_json(jp)
        if data is None:
            continue

        # 取得 lr
        lr_val = find_lr_in_json(data)
        if lr_val is None:
            lr_val = parse_lr_from_path(jp)
        if lr_val is None:
            print(f"[WARN] 無法解析 lr，略過: {jp}")
            continue

        om = get_overall_metrics(data)
        # 修改：支持 train_losses (複數) 和 eval_losses (複數)
        train = fetch_series(om, ["train_losses", "train_loss", "loss_train", "training_loss"]) or []
        evalv = fetch_series(om, ["eval_losses", "eval_loss", "val_loss", "validation_loss", "valid_loss", "loss_val"]) or []
        psnr = fetch_series(om, ["psnr", "psnr_batch", "psnr_val", "val_psnr", "psnr_valid"]) or []
        ssim = fetch_series(om, ["ssim", "ssim_batch", "ssim_val", "val_ssim", "ssim_valid"]) or []

        groups[lr_val]["train_loss"].append(train)
        groups[lr_val]["eval_loss"].append(evalv)
        groups[lr_val]["psnr"].append(psnr)
        groups[lr_val]["ssim"].append(ssim)

        # 紀錄 test psnr（做 lr 選擇）
        tpsnr = None
        tr = data.get("test_results")
        if isinstance(tr, dict):
            # 優先使用 test_psnr
            if isinstance(tr.get("test_psnr"), (int, float)):
                tpsnr = float(tr["test_psnr"])
            elif isinstance(tr.get("psnr_batch"), (int, float)):
                tpsnr = float(tr["psnr_batch"])
            elif isinstance(tr.get("psnr"), (int, float)):
                tpsnr = float(tr["psnr"])
            else:
                ov = tr.get("overall")
                if isinstance(ov, dict):
                    for k in ["psnr_batch", "psnr", "test_psnr"]:
                        if isinstance(ov.get(k), (int, float)):
                            tpsnr = float(ov[k]); break
        if tpsnr is None:
            tpsnr = extract_final_test_psnr(data)
        if tpsnr is not None:
            test_psnr_map[lr_val].append(tpsnr)

    return groups, test_psnr_map

def aggregate_group(group_dict: Dict[str, List[List[float]]]):
    """
    對某 lr 的集合做 mean/std 聚合。
    回傳 dict：
      {
        "train_loss": (x_k, mean, std),
        "eval_loss":  (x_k, mean, std),
        "psnr":       (x_k, mean, std),
        "ssim":       (x_k, mean, std),
      }
    """
    out = {}

    for key, step in [("train_loss", LOSS_STEP), ("eval_loss", METRIC_STEP),
                      ("psnr", METRIC_STEP), ("ssim", METRIC_STEP)]:
        seqs = group_dict.get(key, [])
        # 不做 outlier 過濾，直接使用原始數據
        is_loss = key in ("train_loss", "eval_loss")
        seqs_filtered = [_apply_outlier_mask(seq, is_loss=is_loss) for seq in seqs]

        arr = pad_and_stack(seqs_filtered)  # shape=(runs, T_max)
        mean, std = nanmean_std(arr)
        xk = steps_k(len(mean), step)
        out[key] = (xk, mean, std)
    return out

def select_best_lr(test_psnr_map: Dict[float, List[float]]) -> Optional[float]:
    """
    以 test PSNR 的 '平均' 作為挑選 lr 的準則。
    """
    if not test_psnr_map:
        return None
    best_lr, best_mean = None, None
    for lr, vals in test_psnr_map.items():
        if not vals:
            continue
        m = float(np.mean(vals))
        if (best_mean is None) or (m > best_mean):
            best_mean = m
            best_lr = lr
    return best_lr

# -------------------------- Print Stats --------------------------

def print_metrics_stats(agg_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                        name: str, 
                        save_to_file: Optional[str] = None):
    """打印 PSNR 和 SSIM 的 mean ± std（每個 step）"""
    
    output_lines = []
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"{name} - Metrics Statistics")
    output_lines.append(f"{'='*80}\n")
    
    for metric_name in ["psnr", "ssim"]:
        if metric_name not in agg_dict:
            continue
            
        xk, mean, std = agg_dict[metric_name]
        
        output_lines.append(f"\n{metric_name.upper()}:")
        output_lines.append(f"{'-'*60}")
        output_lines.append(f"{'Step (k)':<12} {'Mean':<15} {'Std':<15} {'Mean±Std':<20}")
        output_lines.append(f"{'-'*60}")
        
        # for i, (step, m, s) in enumerate(zip(xk, mean, std)):
        #     if not (np.isnan(m) or np.isnan(s)):
        #         output_lines.append(
        #             f"{step:>10.2f}   {m:>13.6f}   {s:>13.6f}   {m:.6f}±{s:.6f}"
        #         )
        
        # 打印最終值
        if len(mean) > 0 and not np.isnan(mean[-1]):
            output_lines.append(f"{'-'*60}")
            output_lines.append(
                f"Final: {xk[-1]:.2f}k -> {mean[-1]:.6f} ± {std[-1]:.6f}"
            )
    
    # 輸出到終端
    for line in output_lines:
        print(line)
    
    # 選擇性儲存到檔案
    if save_to_file:
        with open(save_to_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n[INFO] 統計數據已儲存至: {save_to_file}")

# -------------------------- Plot --------------------------
from scipy.signal import savgol_filter

def _maybe_smooth(mean: np.ndarray, do_smooth=True) -> np.ndarray:
    if not do_smooth:
        return mean
    from scipy.signal import savgol_filter
    if len(mean) < 11:
        return mean
    return savgol_filter(mean, window_length=11, polyorder=3)

def plot_aggregated(hg_agg: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    hf_agg: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    title_suffix: str = "",
                    save_path: Optional[str] = None):
    """
    左：上下兩張（上 Train Loss、下 Eval Loss）
    中：PSNR（跨上下）
    右：SSIM（跨上下）
    - x 軸：Step (k)（train_loss 每 500、eval_loss/psnr/ssim 每 2500 的座標已縮放為 k）
    - 不使用 marker；畫均值曲線 + std 陰影
    """
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    # 顏色
    style_cfg = {
        "HG":   dict(color="#0072B2", label="Hourglass"),
        "HG_F": dict(color="#009E73", label=r"Hourglass (fixed $W_{in}$)"),
    }

    def _plot_with_shade(ax, xk, mean, std, color, label, linewidth=2.0):
        if len(xk) == 0:
            return
        ax.plot(xk, mean, linewidth=linewidth, linestyle="-", color=color, label=label)
        ax.fill_between(xk, mean - std, mean + std, alpha=0.18, color=color)

    def _downsample(triple: Tuple[np.ndarray, np.ndarray, np.ndarray], step: int = 2, offset: int = 0):
        """
        從 (xk, mean, std) 抽樣，每 step 個點取 1 個。
        - step=2 → 每2點取1 (等於 odd/even 的一般化)
        - step=5 → 每5點取1
        - offset 決定起始位置 (0=第1點, 1=第2點, ...)
        """
        xk, mean, std = triple
        if len(xk) == 0:
            return triple
        idx = np.arange(offset, len(xk), step, dtype=int)
        return xk[idx], mean[idx], std[idx]

    def _maybe_select(triple: Tuple[np.ndarray, np.ndarray, np.ndarray], kind: str,
        step: int = 2, offset: int = 0, apply_to={"loss","metrics"}):
        if kind in apply_to:
            return _downsample(triple, step=step, offset=offset)
        return triple

    # ---- 上：Train Loss ----
    ax_train = fig.add_subplot(gs[0, 0])
    _plot_with_shade(ax_train, *hg_agg["train_loss"], **style_cfg["HG"])
    _plot_with_shade(ax_train, *hf_agg["train_loss"], **style_cfg["HG_F"])
    ax_train.set_title("Train Loss" + title_suffix, fontsize=18)
    ax_train.set_xlabel("Step (k)", fontsize=14)
    ax_train.grid(True, alpha=0.3)
    ax_train.tick_params(axis='both', which='major', labelsize=13)

    # ---- 下：Eval Loss ----
    ax_eval = fig.add_subplot(gs[1, 0])
    _plot_with_shade(ax_eval, *hg_agg["eval_loss"], **style_cfg["HG"])
    _plot_with_shade(ax_eval, *hf_agg["eval_loss"], **style_cfg["HG_F"])
    ax_eval.set_title("Eval Loss", fontsize=18)
    ax_eval.set_xlabel("Step (k)", fontsize=14)
    ax_eval.grid(True, alpha=0.3)
    ax_eval.tick_params(axis='both', which='major', labelsize=13)

    # 兩張 loss y 軸對齊（可選）
    def _ymax(agg):
        x, m, s = agg
        return np.nanmax(m + s) if len(x) else np.nan
    ymax = np.nanmax([_ymax(hg_agg["train_loss"]), _ymax(hf_agg["train_loss"]),
                      _ymax(hg_agg["eval_loss"]),  _ymax(hf_agg["eval_loss"])])
    if np.isfinite(ymax):
        ymin = 0.0025
        ax_train.set_ylim(ymin, ymax * 1.05)
        ax_eval.set_ylim(ymin, ymax * 1.05)

    # x-ticks：每 5k 一格（loss 與 metrics 都用同一套）
    def _set_ticks_every_5k(ax_list: List[plt.Axes],
                            aggs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        max_x = 0.0
        for (xk, _, _) in aggs:
            if len(xk) > 0:
                max_x = max(max_x, float(xk[-1]))
        if max_x <= 0:
            return
        upper = 5.0 * np.ceil(max_x / 5.0)
        xticks = np.arange(0.0, upper + 5.0, 5.0)
        for a in ax_list:
            a.set_xticks(xticks)

    _set_ticks_every_5k([ax_train, ax_eval],
                        [hg_agg["train_loss"], hf_agg["train_loss"],
                         hg_agg["eval_loss"],  hf_agg["eval_loss"]])

    # ---- 中：PSNR（跨上下）----
    ax = fig.add_subplot(gs[:, 1])
    _plot_with_shade(ax, *hg_agg["psnr"], **style_cfg["HG"])
    _plot_with_shade(ax, *hf_agg["psnr"], **style_cfg["HG_F"])
    ax.set_title("PSNR", fontsize=18)
    ax.set_xlabel("Step (k)", fontsize=14)
    # ax.set_ylim(19.0, 22.3)
    # ax.set_ylim(16.5, 23.0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)

    # ---- 右：SSIM（跨上下）----
    ax = fig.add_subplot(gs[:, 2])
    _plot_with_shade(ax, *hg_agg["ssim"], **style_cfg["HG"])
    _plot_with_shade(ax, *hf_agg["ssim"], **style_cfg["HG_F"])
    ax.set_title("SSIM", fontsize=18)
    ax.set_xlabel("Step (k)", fontsize=14)
    # ax.set_ylim(0.55, 0.70)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)

    _set_ticks_every_5k([fig.axes[2], fig.axes[3]],
                        [hg_agg["psnr"], hf_agg["psnr"],
                         hg_agg["ssim"], hf_agg["ssim"]])

    # 單一總 legend（放中上方）
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    if handles:
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        fig.legend(uniq.values(), uniq.keys(),
                   loc="upper center", ncol=2, frameon=True, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] 圖片已儲存至: {save_path}")

# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-run (per-lr) results with mean±std and plot")
    parser.add_argument('--hourglass_path', type=str, required=True,
                        help="Hourglass 總資料夾（底下含 lrX_runY 子資料夾）")
    parser.add_argument('--hourglass_fz_path', type=str, required=True,
                        help="Hourglass-Freeze 總資料夾（底下含 lrX_runY 子資料夾）")
    parser.add_argument('--save_stats', type=str, default=None,
                        help="儲存統計數據的文字檔路徑")
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    # 先掃一遍（可選，保留原來單 run best 日誌）
    pick_best_experiment(args.hourglass_path)
    pick_best_experiment(args.hourglass_fz_path)

    # 分群（依 lr）並聚合
    hg_groups, hg_test_map = group_runs_by_lr(args.hourglass_path)
    hf_groups, hf_test_map = group_runs_by_lr(args.hourglass_fz_path)

    if not hg_groups:
        print("[ERROR] Hourglass 沒有可用的組（無法解析 lr 或無 JSON）")
        return
    if not hf_groups:
        print("[ERROR] Hourglass-Freeze 沒有可用的組（無法解析 lr 或無 JSON）")
        return

    # 以 test PSNR 的平均挑出每邊最佳 lr
    hg_best_lr = select_best_lr(hg_test_map)
    hf_best_lr = select_best_lr(hf_test_map)

    if hg_best_lr is None:
        print("[ERROR] Hourglass 無 test PSNR 可挑最佳 lr")
        return
    if hf_best_lr is None:
        print("[ERROR] Hourglass-Freeze 無 test PSNR 可挑最佳 lr")
        return

    print(f"[INFO] Hourglass 最佳 lr: {fmt_lr(hg_best_lr)} (基於 test PSNR 平均)")
    print(f"[INFO] Hourglass-Freeze 最佳 lr: {fmt_lr(hf_best_lr)} (基於 test PSNR 平均)")

    # 聚合（mean/std，含離群點過濾）
    hg_agg = aggregate_group(hg_groups[hg_best_lr])
    hf_agg = aggregate_group(hf_groups[hf_best_lr])
    
    print_metrics_stats(hg_agg, f"Hourglass (lr={fmt_lr(hg_best_lr)})", 
                    save_to_file=args.save_stats.replace('.txt', '_HG.txt') if args.save_stats else None)
    print_metrics_stats(hf_agg, f"Hourglass-Freeze (lr={fmt_lr(hf_best_lr)})", 
                    save_to_file=args.save_stats.replace('.txt', '_HGF.txt') if args.save_stats else None)

    # 畫圖
    title_suffix = ""
    plot_aggregated(hg_agg, hf_agg, title_suffix, args.save_path)

if __name__ == "__main__":
    main()