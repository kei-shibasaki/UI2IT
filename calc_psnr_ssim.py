import os 
import glob 
import numpy as np 
import cv2 
from tqdm import tqdm

from scripts.metrics import calculate_psnr, calculate_ssim

names = [
    'MSPC_edges2shoes_b04', 
    'ours_edges2shoes'
]

dataset_name = 'edges2shoes'
dirsA = [f'experiments/{name}/generated' for name in names]
dirsB = [f'datasets/{dataset_name}/testA'] * len(names)


for name, dirA, dirB in zip(names, dirsA, dirsB):
    with open(f'psnr_ssim_{name}.csv', 'w', encoding='utf-8') as fp:
        fp.write('step,psnr,ssim\n')

    imagesB = sorted(glob.glob(os.path.join(dirB, '*.jpg')))
    
    subdirsA = [d for d in sorted(glob.glob(os.path.join(dirA, '*'))) if os.path.isdir(d)]
    steps = [os.path.basename(subdirA) for subdirA in subdirsA]
    subdirsA = [os.path.join(subdirA, 'GA') for subdirA in subdirsA]

    for step, subdirA in zip(steps, subdirsA):
        imagesA = sorted(glob.glob(os.path.join(subdirA, '*.png')))
        psnrs = []
        ssims = []
        for img_pathA, img_pathB in zip(imagesA, imagesB):
            imgA = cv2.imread(img_pathA)
            imgB = cv2.imread(img_pathB)

            psnr = calculate_psnr(imgA, imgB, crop_border=0)
            ssim = calculate_ssim(imgA, imgB, crop_border=0)
            psnrs.append(psnr)
            ssims.append(ssim)
        
        psnr_mean = sum(psnrs) / len(psnrs)
        ssim_mean = sum(ssims) / len(ssims)
        with open(f'psnr_ssim_{name}.csv', 'a', encoding='utf-8') as fp:
            fp.write(f'{int(step)},{psnr_mean:.2f},{ssim_mean:.3f}\n')

        print(f'{subdirA}, PSNR: {psnr_mean:.2f}, SSIM: {ssim_mean:.3f}')

