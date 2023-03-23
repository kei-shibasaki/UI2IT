import glob
import os
import random

import torch
import torch.utils.data
import torchvision
from torchvision.transforms import functional as TF
from tqdm import tqdm
import cv2 

from scripts.utils import load_img_uint8, read_img

class UnpairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, target_path, input_resolution, ext='jpg', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths_A = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.paths_B = sorted(glob.glob(os.path.join(target_path, f'*.{ext}')))

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images_A = [self.to_cache(img_path) for img_path in self.paths_A]
            self.images_B = [self.to_cache(img_path) for img_path in self.paths_B]
        else:
            self.images_A = None
            self.images_B = None
    
    def to_cache(self, img_path):
        img = load_img_uint8(img_path, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        # Randomly select images
        idx_A = random.randint(0, len(self.paths_A)-1)
        idx_B = random.randint(0, len(self.paths_B)-1)
        if self.cache_images:
            img_A = self.images_A[idx_A].to(torch.float32) / 255.0
            img_B = self.images_B[idx_B].to(torch.float32) / 255.0
        else:
            img_A = TF.resize(read_img(self.paths_A[idx_A]), [self.h, self.w], self.inter_mode)
            img_B = TF.resize(read_img(self.paths_B[idx_B]), [self.h, self.w], self.inter_mode)
        
        return {'A': img_A, 'B': img_B}
    
    def __len__(self):
        return min(len(self.paths_A), len(self.paths_B))

class SimgleImageDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, input_resolution, ext='jpg', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images = [self.to_cache(img_path) for img_path in self.paths]
        else:
            self.images = None
    
    def to_cache(self, img_path):
        img = load_img_uint8(img_path, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        if self.cache_images:
            img = self.images[idx].to(torch.float32) / 255.0
        else:
            img = TF.resize(read_img(self.paths[idx]), [self.h, self.w], self.inter_mode)
        
        return img
    
    def __len__(self):
        return len(self.paths)