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
            self.images_A = [self.to_cache(img_path) for img_path in tqdm(self.paths_A)]
            self.images_B = [self.to_cache(img_path) for img_path in tqdm(self.paths_B)]
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

class UnpairedMaskDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, target_path, input_resolution, ext='png', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths_A = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.paths_B = sorted(glob.glob(os.path.join(target_path, f'*.{ext}')))

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images_A = [self.to_cache(img_path, is_gray=True) for img_path in tqdm(self.paths_A)]
            self.images_B = [self.to_cache(img_path, is_gray=True) for img_path in tqdm(self.paths_B)]
        else:
            self.images_A = None
            self.images_B = None
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, is_gray, to_tensor=True).permute(2,0,1)
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
            img_A = TF.resize(read_img(self.paths_A[idx_A], is_gray=True), [self.h, self.w], self.inter_mode)
            img_B = TF.resize(read_img(self.paths_B[idx_B], is_gray=True), [self.h, self.w], self.inter_mode)
        
        return {'A': img_A, 'B': img_B}
    
    def __len__(self):
        return min(len(self.paths_A), len(self.paths_B))

class SingleImageDataset(torch.utils.data.Dataset):
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
    
class SingleMaskDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, input_resolution, ext='png', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images = [self.to_cache(img_path, is_gray=True) for img_path in self.paths]
        else:
            self.images = None
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, is_gray, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        if self.cache_images:
            img = self.images[idx].to(torch.float32) / 255.0
        else:
            img = TF.resize(read_img(self.paths[idx], is_gray=True), [self.h, self.w], self.inter_mode)
        
        return img
    
    def __len__(self):
        return len(self.paths)

class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, target_path, input_resolution, ext='jpg', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths_A = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.paths_B = sorted(glob.glob(os.path.join(target_path, f'*.{ext}')))
        assert len(self.paths_A)==len(self.paths_B)

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images_A = [self.to_cache(img_path) for img_path in self.paths_A]
            self.images_B = [self.to_cache(img_path) for img_path in self.paths_B]
        else:
            self.images_A = None
            self.images_B = None
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        if self.cache_images:
            img_A = self.images_A[idx].to(torch.float32) / 255.0
            img_B = self.images_B[idx].to(torch.float32) / 255.0
        else:
            img_A = TF.resize(read_img(self.paths_A[idx]), [self.h, self.w], self.inter_mode)
            img_B = TF.resize(read_img(self.paths_B[idx]), [self.h, self.w], self.inter_mode)
        
        return {'A': img_A, 'B': img_B}
    
    def __len__(self):
        return min(len(self.paths_A), len(self.paths_B))

class SingleImageDatasetWithMask(torch.utils.data.Dataset):
    def __init__(self, source_path, source_mask_path, input_resolution, ext='jpg', ext_mask='png', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.mask_path = sorted(glob.glob(os.path.join(source_mask_path, f'*.{ext_mask}')))
        assert len(self.paths)==len(self.mask_path)
        
        # Save 8uint training images into memory.
        if self.cache_images:
            self.images = [self.to_cache(img_path) for img_path in tqdm(self.paths)]
            self.masks = [self.to_cache(mask_path, is_gray=True) for mask_path in tqdm(self.mask_path)]
        else:
            self.images = None
            self.masks = None 
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, is_gray, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        if self.cache_images:
            img = self.images[idx].to(torch.float32) / 255.0
            mask = self.masks[idx].to(torch.float32) / 255.0
        else:
            img = TF.resize(read_img(self.paths[idx]), [self.h, self.w], self.inter_mode)
            mask = TF.resize(read_img(self.mask_path[idx], is_gray=True), [self.h, self.w], self.inter_mode)
        
        return {'img': img, 'mask': mask}
    
    def __len__(self):
        return len(self.paths)

class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, input_resolution, ext='jpg', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        
        # Save 8uint training images into memory.
        if self.cache_images:
            self.images = [self.to_cache(img_path) for img_path in tqdm(self.paths)]
        else:
            self.images = None
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, is_gray, to_tensor=True).permute(2,0,1)
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

class UnpairedImageDatasetWithMask(torch.utils.data.Dataset):
    def __init__(self, source_path, source_mask_path, target_path, input_resolution, ext='jpg', ext_mask='png', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths_A = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.paths_B = sorted(glob.glob(os.path.join(target_path, f'*.{ext}')))

        self.paths_A_mask = sorted(glob.glob(os.path.join(source_mask_path, f'*.{ext_mask}')))

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images_A = [self.to_cache(img_path) for img_path in tqdm(self.paths_A)]
            self.images_B = [self.to_cache(img_path) for img_path in tqdm(self.paths_B)]
            self.masks_A = [self.to_cache(mask_path, is_gray=True) for mask_path in tqdm(self.paths_A_mask)]
        else:
            self.images_A = None
            self.images_B = None
            self.masks_A = None
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, is_gray, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        # Randomly select images
        idx_A = random.randint(0, len(self.paths_A)-1)
        idx_B = random.randint(0, len(self.paths_B)-1)
        if self.cache_images:
            img_A = self.images_A[idx_A].to(torch.float32) / 255.0
            img_B = self.images_B[idx_B].to(torch.float32) / 255.0
            mask_A = self.masks_A[idx_A].to(torch.float32) / 255.0
        else:
            img_A = TF.resize(read_img(self.paths_A[idx_A]), [self.h, self.w], self.inter_mode)
            img_B = TF.resize(read_img(self.paths_B[idx_B]), [self.h, self.w], self.inter_mode)
            mask_A = TF.resize(read_img(self.paths_A_mask[idx_A], is_gray=True), [self.h, self.w], self.inter_mode)
        
        return {'A': img_A, 'B': img_B, 'mask_A': mask_A}
    
    def __len__(self):
        return min(len(self.paths_A), len(self.paths_B))
    

class UnpairedImageDatasetWithMask(torch.utils.data.Dataset):
    def __init__(self, source_path, source_mask_path, target_path, target_mask_path, input_resolution, ext='jpg', ext_mask='png', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths_A = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.paths_B = sorted(glob.glob(os.path.join(target_path, f'*.{ext}')))

        self.paths_A_mask = sorted(glob.glob(os.path.join(source_mask_path, f'*.{ext_mask}')))
        self.paths_B_mask = sorted(glob.glob(os.path.join(target_mask_path, f'*.{ext_mask}')))

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images_A = [self.to_cache(img_path) for img_path in tqdm(self.paths_A)]
            self.images_B = [self.to_cache(img_path) for img_path in tqdm(self.paths_B)]
            self.masks_A = [self.to_cache(mask_path, is_gray=True) for mask_path in tqdm(self.paths_A_mask)]
            self.masks_B = [self.to_cache(mask_path, is_gray=True) for mask_path in tqdm(self.paths_B_mask)]
        else:
            self.images_A = None
            self.images_B = None
            self.masks_A = None
            self.masks_B = None
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, is_gray, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        # Randomly select images
        idx_A = random.randint(0, len(self.paths_A)-1)
        idx_B = random.randint(0, len(self.paths_B)-1)
        if self.cache_images:
            img_A = self.images_A[idx_A].to(torch.float32) / 255.0
            img_B = self.images_B[idx_B].to(torch.float32) / 255.0
            mask_A = self.masks_A[idx_A].to(torch.float32) / 255.0
            mask_B = self.masks_B[idx_B].to(torch.float32) / 255.0
        else:
            img_A = TF.resize(read_img(self.paths_A[idx_A]), [self.h, self.w], self.inter_mode)
            img_B = TF.resize(read_img(self.paths_B[idx_B]), [self.h, self.w], self.inter_mode)
            mask_A = TF.resize(read_img(self.paths_A_mask[idx_A], is_gray=True), [self.h, self.w], self.inter_mode)
            mask_B = TF.resize(read_img(self.paths_B_mask[idx_B], is_gray=True), [self.h, self.w], self.inter_mode)
        
        return {'A': img_A, 'B': img_B, 'mask_A': mask_A, 'mask_B': mask_B}
    
    def __len__(self):
        return min(len(self.paths_A), len(self.paths_B))

class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, target_path, input_resolution, ext='jpg', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths_A = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.paths_B = sorted(glob.glob(os.path.join(target_path, f'*.{ext}')))

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images_A = [self.to_cache(img_path) for img_path in tqdm(self.paths_A)]
            self.images_B = [self.to_cache(img_path) for img_path in tqdm(self.paths_B)]
        else:
            self.images_A = None
            self.images_B = None
    
    def to_cache(self, img_path):
        img = load_img_uint8(img_path, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        if self.cache_images:
            img_A = self.images_A[idx].to(torch.float32) / 255.0
            img_B = self.images_B[idx].to(torch.float32) / 255.0
        else:
            img_A = TF.resize(read_img(self.paths_A[idx]), [self.h, self.w], self.inter_mode)
            img_B = TF.resize(read_img(self.paths_B[idx]), [self.h, self.w], self.inter_mode)
        
        return {'A': img_A, 'B': img_B}
    
    def __len__(self):
        return min(len(self.paths_A), len(self.paths_B))


"""
class SingleImageDatasetWithMask(torch.utils.data.Dataset):
    def __init__(self, source_path, source_mask_paths, input_resolution, ext='jpg', ext_mask='png', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.mask_paths = []
        for mask_path in source_mask_paths:
            self.mask_paths.append(sorted(glob.glob(os.path.join(mask_path, f'*.{ext_mask}'))))
        
        # Save 8uint training images into memory.
        if self.cache_images:
            self.images = [self.to_cache(img_path) for img_path in tqdm(self.paths)]
            self.masks = []
            for temp in self.mask_paths:
                self.masks.append([self.to_cache(mask_path, is_gray=True) for mask_path in tqdm(temp)])
        else:
            self.images = None
            self.masks = None 
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, is_gray, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        if self.cache_images:
            img = self.images[idx].to(torch.float32) / 255.0
            masks = []
            for mask_list in self.masks:
                masks.append(mask_list[idx].to(torch.float32) / 255.0)
        else:
            img = TF.resize(read_img(self.paths[idx]), [self.h, self.w], self.inter_mode)
            masks = []
            for mask_path in self.mask_paths:
                masks.append(TF.resize(read_img(mask_path[idx], is_gray=True), [self.h, self.w], self.inter_mode))
        
        return {'img': img, 'masks': masks}
    
    def __len__(self):
        return len(self.paths)

class UnpairedImageDatasetWithMask(torch.utils.data.Dataset):
    def __init__(self, source_path, source_mask_paths, target_path, input_resolution, ext='jpg', ext_mask='png', cache_images=True):
        super().__init__()
        self.h, self.w = input_resolution
        self.cache_images = cache_images
        self.inter_mode = TF.InterpolationMode.BILINEAR
        self.paths_A = sorted(glob.glob(os.path.join(source_path, f'*.{ext}')))
        self.paths_B = sorted(glob.glob(os.path.join(target_path, f'*.{ext}')))

        self.paths_A_mask = []
        for mask_path in source_mask_paths:
            self.paths_A_mask.append(sorted(glob.glob(os.path.join(mask_path, f'*.{ext_mask}'))))

        # Save 8uint training images into memory.
        if self.cache_images:
            self.images_A = [self.to_cache(img_path) for img_path in tqdm(self.paths_A)]
            self.images_B = [self.to_cache(img_path) for img_path in tqdm(self.paths_B)]
            self.masks_A = []
            for temp in self.paths_A_mask:
                self.masks_A.append([self.to_cache(mask_path, is_gray=True) for mask_path in tqdm(temp)])
        else:
            self.images_A = None
            self.images_B = None    
            self.masks_A = None
    
    def to_cache(self, img_path, is_gray=False):
        img = load_img_uint8(img_path, is_gray, to_tensor=True).permute(2,0,1)
        img = TF.resize(img, [self.h, self.w], self.inter_mode)
        return img
    
    def __getitem__(self, idx):
        # Randomly select images
        idx_A = random.randint(0, len(self.paths_A)-1)
        idx_B = random.randint(0, len(self.paths_B)-1)
        if self.cache_images:
            img_A = self.images_A[idx_A].to(torch.float32) / 255.0
            img_B = self.images_B[idx_B].to(torch.float32) / 255.0
            masks_A = []
            for mask_list in self.masks_A:
                masks_A.append(mask_list[idx_A].to(torch.float32) / 255.0)
        else:
            img_A = TF.resize(read_img(self.paths_A[idx_A]), [self.h, self.w], self.inter_mode)
            img_B = TF.resize(read_img(self.paths_B[idx_B]), [self.h, self.w], self.inter_mode)
            masks_A = []
            for mask_path in self.paths_A_mask:
                masks_A.append(TF.resize(read_img(mask_path[idx_A], is_gray=True), [self.h, self.w], self.inter_mode))
        
        return {'A': img_A, 'B': img_B, 'masks_A': masks_A}
    
    def __len__(self):
        return min(len(self.paths_A), len(self.paths_B))
"""