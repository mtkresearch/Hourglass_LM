import torch
import pickle
import numpy as np
import os
import sys

# 載入 cache
cache_dir = './old/data/denoising_cache/imagenet32_noise0.25_eval'
noisy_file = os.path.join(cache_dir, 'noisy_data.pkl')
index_file = os.path.join(cache_dir, 'index_mapping.pkl')

print("=" * 60)
print("載入 Cache 檔案...")
print("=" * 60)

with open(noisy_file, 'rb') as f:
    noisy_images = pickle.load(f)

with open(index_file, 'rb') as f:
    index_mapping = pickle.load(f)

print(f"✓ 載入 {len(noisy_images)} 個 noisy images")
print(f"✓ 載入 {len(index_mapping)} 個 index mappings")
print()

# 檢查基本一致性
assert len(noisy_images) == len(index_mapping), "數量不匹配！"

print("=" * 60)
print("檢查 Index Mapping 結構...")
print("=" * 60)
print(f"前 5 個 mapping: {index_mapping[:5]}")
print()

# 檢查 original_idx 是否連續
original_indices = [m[0] for m in index_mapping]
print(f"Original indices 範圍: {min(original_indices)} ~ {max(original_indices)}")
print(f"是否連續: {original_indices == list(range(len(original_indices)))}")
print()

# 現在需要載入原始 ImageNet-32 dataset 來驗證
print("=" * 60)
print("載入原始 ImageNet-32 Dataset...")
print("=" * 60)

# 需要知道 ImageNet-32 的路徑
imagenet32_root = './old/data/ImageNet-32-source/val'  # 假設路徑

# 載入 unpickle 函數
def unpickle(file):
    try:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
    except UnicodeDecodeError:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
    return dict

try:
    val_file = os.path.join(imagenet32_root, 'val_data')
    
    if os.path.exists(val_file):
        print(f"✓ 找到 val_data: {val_file}")
        d = unpickle(val_file)
        x = d['data'].astype(np.float32) / 255.0
        y = np.array([i-1 for i in d['labels']], dtype=np.int64)
        
        # 重塑為 (N, 3, 32, 32)
        img_size = 32
        img_size2 = img_size * img_size
        
        print("正在重塑圖片...")
        x_reshaped = []
        for j in range(x.shape[0]):
            single_img = x[j]
            r = single_img[:img_size2].reshape(img_size, img_size)
            g = single_img[img_size2:2*img_size2].reshape(img_size, img_size)
            b = single_img[2*img_size2:].reshape(img_size, img_size)
            rgb_img = np.stack([r, g, b], axis=0)
            x_reshaped.append(rgb_img)
        
        original_data = np.array(x_reshaped)
        original_labels = y
        print(f"✓ 載入 {len(original_data)} 張原始圖片")
        print()
        
        # 建立 eval split (需要跟 Document 1 一樣的方式切分)
        print("=" * 60)
        print("建立 Eval Split (模擬 Document 1 的切分方式)...")
        print("=" * 60)
        
        total_size = len(original_data)
        local_rng = np.random.RandomState(42)  # 固定 seed
        indices = local_rng.permutation(total_size)
        half_size = total_size // 2
        
        eval_indices = indices[:half_size]
        print(f"✓ Eval split: {len(eval_indices)} 張圖片")
        print(f"  Eval indices 範圍: {eval_indices.min()} ~ {eval_indices.max()}")
        print()
        
        # 驗證 Mapping
        print("=" * 60)
        print("驗證 Cache Mapping (檢查前 10 張)...")
        print("=" * 60)
        
        noise_std = 0.25
        all_correct = True
        
        for idx in range(min(10, len(noisy_images))):
            # 從 cache 讀取
            noisy_cached = torch.FloatTensor(noisy_images[idx])
            original_idx, label_cached, aug_type = index_mapping[idx]
            
            # 從原始資料讀取 (透過 eval_indices)
            actual_data_idx = eval_indices[original_idx]
            clean_img = torch.FloatTensor(original_data[actual_data_idx])
            label_actual = original_labels[actual_data_idx]
            
            # 計算噪聲
            noise = noisy_cached - clean_img
            
            # 檢查
            noise_mean = noise.mean().item()
            noise_std_actual = noise.std().item()
            
            print(f"\nSample {idx}:")
            print(f"  Cache idx: {idx} -> Original idx: {original_idx} -> Actual data idx: {actual_data_idx}")
            print(f"  Label (cached): {label_cached}, Label (actual): {label_actual}")
            print(f"  Noise mean: {noise_mean:.6f} (期望 ~0)")
            print(f"  Noise std: {noise_std_actual:.6f} (期望 ~{noise_std})")
            print(f"  Clean image range: [{clean_img.min():.3f}, {clean_img.max():.3f}]")
            print(f"  Noisy image range: [{noisy_cached.min():.3f}, {noisy_cached.max():.3f}]")
            
            # 檢查是否合理
            if label_cached != label_actual:
                print(f"  ❌ LABEL MISMATCH!")
                all_correct = False
            if abs(noise_mean) > 0.05:
                print(f"  ⚠️  Noise mean 偏離 0 較多")
            if abs(noise_std_actual - noise_std) > 0.05:
                print(f"  ⚠️  Noise std 偏離預期較多")
        
        print("\n" + "=" * 60)
        if all_correct:
            print("✅ 前 10 個樣本的 Mapping 都正確！")
        else:
            print("❌ 發現 Mapping 錯誤！")
        print("=" * 60)
        
    else:
        print(f"❌ 找不到檔案: {val_file}")
        print("請修改程式中的 imagenet32_root 路徑")
        
except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()