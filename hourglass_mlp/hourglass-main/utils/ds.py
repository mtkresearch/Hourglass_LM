import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset
from PIL import Image
import os
import pickle
import numpy as np


class ImageNet32Dataset(Dataset):
    def __init__(self, data_folder, split="train", transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        if split == "train":
            print("Loading ImageNet-32 training data...")
            for i in range(1, 11):
                batch_file = os.path.join(data_folder, f'train_data_batch_{i}')
                if os.path.exists(batch_file):
                    print(f"  Loading batch {i}...")
                    d = self.unpickle(batch_file)
                    x = d['data'].astype(np.float32) / 255.0
                    y = np.array([i-1 for i in d['labels']], dtype=np.int64)

                    img_size = 32
                    img_size2 = img_size * img_size
                    x_reshaped = []
                    
                    for j in range(x.shape[0]):
                        single_img = x[j]
                        r = single_img[:img_size2].reshape(img_size, img_size)
                        g = single_img[img_size2:2*img_size2].reshape(img_size, img_size)
                        b = single_img[2*img_size2:].reshape(img_size, img_size)
                        rgb_img = np.stack([r, g, b], axis=0)
                        x_reshaped.append(rgb_img)                    
                    x_reshaped = np.array(x_reshaped)
                    self.data.append(x_reshaped)
                    self.labels.append(y)
            self.data = np.concatenate(self.data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            
        elif split in ["eval", "test"]:
            print("Loading ImageNet-32 validation data...")
            val_file = os.path.join(data_folder, 'val_data')
            if os.path.exists(val_file):
                d = self.unpickle(val_file)
                x = d['data'].astype(np.float32) / 255.0
                y = np.array([i-1 for i in d['labels']], dtype=np.int64)
                
                img_size = 32
                img_size2 = img_size * img_size
                
                x_reshaped = []
                for j in range(x.shape[0]):
                    single_img = x[j]
                    r = single_img[:img_size2].reshape(img_size, img_size)
                    g = single_img[img_size2:2*img_size2].reshape(img_size, img_size)
                    b = single_img[2*img_size2:].reshape(img_size, img_size)
                    rgb_img = np.stack([r, g, b], axis=0)
                    x_reshaped.append(rgb_img)
                
                self.data = np.array(x_reshaped)
                self.labels = y
            else:
                raise FileNotFoundError(f"Validation file not found: {val_file}")
        else:
            raise ValueError(f"Unsupported split: {split}. Must be one of ['train', 'val', 'eval', 'test']")
        
        print(f"ImageNet-32 {split} loaded: {len(self.data)} images, shape: {self.data.shape}")
    
    def unpickle(self, file):
        try:
            with open(file, 'rb') as fo:
                dict = pickle.load(fo)
        except UnicodeDecodeError:
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='latin1')
        return dict

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.data[idx])  # (3, 32, 32)
        label = int(self.labels[idx])
        
        if self.transform:
            image_hwc = image.permute(1, 2, 0)
            image_hwc = self.transform(image_hwc)
            if len(image_hwc.shape) == 3:
                image = image_hwc.permute(2, 0, 1)
            else:
                image = image_hwc
        return image, label



class PairDataset(Dataset):
    def __init__(self, dataset_name='mnist', root='./data', 
                 split='train', mode=None,
                 noise_std=0.0, down_scale=2.0, 
                 use_augmentation=False, aug_num=None,
                 random_seed=42):
        self.mode = mode
        self.noise_std = noise_std
        self.down_scale = down_scale
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.split = split
        self.random_seed = random_seed
        self.use_augmentation = use_augmentation
        self.aug_num = aug_num

        if self.split not in ['train', 'eval', 'test']:
            raise ValueError(f"split must be one of ['train', 'eval', 'test'], got {split}")
        
        if self.dataset_name not in ['mnist', 'imagenet32']:
            raise NotImplementedError(f"Dataset currectly only support for MNIST and ImageNet-32, got {dataset_name}")
        
        if self.mode == 'denoising':
            assert self.noise_std != 0.0, "noise_std must be non-zero for denoising mode"
        elif self.mode == 'super_resolution':
            assert self.down_scale > 1.0, "down_scale must be > 1.0 for super_resolution mode"

        # Create correct data split
        transform = T.Compose([T.ToTensor()])
        if self.dataset_name == 'mnist':
            if self.split == 'test':
                self.dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
            elif self.split in ['train', 'eval']:
                full_train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
                self.dataset = self._create_train_eval_split(full_train_dataset)               
        elif self.dataset_name == 'imagenet32':
            imagenet32_root = os.path.join(root, "ImageNet-32")    
            if self.split == 'train':
                train_folder = os.path.join(imagenet32_root, "train")
                self.dataset = ImageNet32Dataset(train_folder, split="train", transform=None)
            elif self.split in ['eval', 'test']:
                val_folder = os.path.join(imagenet32_root, "val") 
                full_val_dataset = ImageNet32Dataset(val_folder, split="eval", transform=None)
                self.dataset = self._create_imagenet32_eval_test_split(full_val_dataset)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Create prototype images for generative classification
        if self.mode == 'generative_classification':
            self.prototypes = {}
            if self.dataset_name == 'mnist':
                for label in range(10):
                    for img, lbl in self.dataset:
                        if lbl == label:
                            self.prototypes[label] = img.clone()
                            break
            else:
                raise NotImplementedError()
    
        sample_img, _ = self.dataset[0]
        self.C, self.H, self.W = sample_img.shape
        self.input_dim = self.C * self.H * self.W
    
    def _create_train_eval_split(self, full_dataset):
        total_size = len(full_dataset)
        local_rng = np.random.RandomState(self.random_seed)
        indices = local_rng.permutation(total_size)
        
        if self.dataset_name == 'mnist':
            if self.split == 'train':
                selected_indices = indices[:50000]
                print(f"Created train split with {len(selected_indices)} samples")
            elif self.split == 'eval':
                selected_indices = indices[50000:]
                print(f"Created eval split with {len(selected_indices)} samples")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return Subset(full_dataset, selected_indices)
    
    def _create_imagenet32_eval_test_split(self, full_val_dataset):
        total_size = len(full_val_dataset)
        local_rng = np.random.RandomState(self.random_seed)
        indices = local_rng.permutation(total_size)
        
        half_size = total_size // 2
        
        if self.split == 'eval':
            selected_indices = indices[:half_size]
            print(f"Created ImageNet-32 eval split with {len(selected_indices)} samples")
        elif self.split == 'test':
            selected_indices = indices[half_size:]
            print(f"Created ImageNet-32 test split with {len(selected_indices)} samples")
        
        return Subset(full_val_dataset, selected_indices)

    def _downsample_image(self, img):
        C, H, W = img.shape
        target_H = H // int(self.down_scale)
        target_W = W // int(self.down_scale)
        img_batch = img.unsqueeze(0)  # (1,C,H,W)
        downsampled = F.interpolate(
            img_batch, 
            size=(target_H, target_W),
            mode='bicubic',
            align_corners=False,
            antialias=False
        )
        return downsampled.squeeze(0)

    def _add_noise_to_image(self, img):
        noise = torch.randn_like(img) * self.noise_std
        noisy_img = torch.clamp(img + noise, 0., 1.)
        return noisy_img

    def _apply_augmentation(self, img, aug_type=None):
        if aug_type == 1:
            return torch.flip(img, dims=[2]) # horizontal flip
        elif aug_type == 2:
            return torch.flip(img, dims=[1]) # vertical flip
        elif aug_type == 3:
            return torch.flip(torch.flip(img, dims=[2]), dims=[1])  # horizontal flip, followed by vertical flip
        else:
            raise ValueError(f"Unsupported aug_type: {aug_type}, must be one of [1, 2, 3]")
 
    def __len__(self):
        base_len = len(self.dataset)
        if self.use_augmentation:
            if self.aug_num in [2, 4]:
                return base_len * self.aug_num
            else:
                raise ValueError(f"aug_num must be 2 or 4 when use_augmentation=True, got {self.aug_num}")
        return base_len


    @torch.no_grad()
    def __getitem__(self, idx):
        aug_type = 0         # 0: no aug, 1: hflip, 2: vflip, 3: both
        original_idx = idx

        if self.use_augmentation:
            if self.mode in ['super_resolution', 'denoising', 'generative_classification']:
                dataset_size = len(self.dataset)
                if self.aug_num == 4:
                    fold = idx // dataset_size            # 0..3
                    original_idx = idx % dataset_size
                    aug_type = fold                       # 0->none, 1->h, 2->v, 3->both
                elif self.aug_num == 2:
                    fold = idx // dataset_size            # 0..1
                    original_idx = idx % dataset_size
                    aug_type = 0 if fold == 0 else 3      # none / both
                else:
                    raise ValueError(f"aug_num must be 2 or 4 when use_augmentation=True, got {self.aug_num}")
            else:
                raise NotImplementedError(f'Unsupported mode ({self.mode}) for data augmentation')

        img, label = self.dataset[original_idx]
        if aug_type != 0:
            img = self._apply_augmentation(img, aug_type=aug_type)

        if self.mode == 'generative_classification':
            noisy_img  = img.clone()
            target_img = self.prototypes[label].clone()
            input_img  = noisy_img
        elif self.mode == 'super_resolution':
            target_img = img.clone()                  # High Resolution
            input_img  = self._downsample_image(img)  # Low Resolution
        elif self.mode == 'denoising':
            target_img = img.clone()
            input_img  = self._add_noise_to_image(img)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return input_img.view(-1), target_img.view(-1)
