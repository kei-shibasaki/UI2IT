import os 
import glob 
import numpy as np 
import cv2 
from tqdm import tqdm 
from PIL import Image

from scripts.utils import arrange_images, read_img

def multiply_mask(img, mask):
    mask = np.expand_dims(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), -1)
    return img*(mask/255.0)

def get_images(images_dir):
    extentions = ['jpg', 'png']
    images = []
    for ext in extentions:
        images = images + glob.glob(os.path.join(images_dir, f'*.{ext}'))
    images.sort()
    return images

def compare_images(images_dirs, out_dir):
    images_list = [get_images(images_dir) for images_dir in images_dirs]

    n_images = len(images_list[0])
    n_row = len(images_list)
    #print(n_images, n_row)

    for i in tqdm(range(n_images)):
        images_path = [images_list[j][i] for j in range(n_row)]
        images = [cv2.imread(img_path) for img_path in images_path]
        #images[1] = multiply_mask(images[0], images[1])
        #images[2] = multiply_mask(images[0], images[2])
        #images[3] = multiply_mask(images[0], images[3])
        #images[4] = multiply_mask(images[0], images[4])
        compare = np.hstack(images)
        cv2.imwrite(os.path.join(out_dir, f'{i:03}.jpg'), compare)



if __name__=='__main__':
    images_dirs = [
        'experiments/TEST_ours_anime_sal2_lr/generated/680000/A',
        '/home/shibasaki/MyTask/pytorch-CycleGAN-and-pix2pix/results_cycle_selfie2anime_cycle/fake_B',
        'results_attentionganv2/selfie2anime/GA', 
        'experiments/MSPC_paper_anime/generated/198000/GA',
        'experiments/TEST_ours_anime_sal2_lr/generated/680000/GA', 
        'experiments/TEST_ours_anime_saladv_lr_sep_recons_clip_idt_mod/generated/700000/GA', 
        'experiments/TEST_ours_anime_saladv_lr_sep_recons_clip_idt_mod/generated/700000/M_A', 
        'experiments/TEST_ours_anime_saladv_lr_sep_recons_clip_idt_mod/generated/700000/TM_GA', 
    ]

    images_dirs = [
        'experiments/ours_horse2zebra_lr_sep_recons_idt/generated/220000/A',
        'experiments/ours_horse2zebra_lr_sep_recons_idt/generated/220000/GA',
        'experiments/ours_horse2zebra_lr_sep_recons10_idt/generated/220000/GA',
        'experiments/ours_horse2zebra_lr_sep_recons_idt10/generated/220000/GA',
        'experiments/ours_horse2zebra_lr_sep_recons10_idt10/generated/220000/GA',
    ]

    images_dirs = [
        'datasets/cat2dog/testA',
        'datasets/cat2dog/testA_map',
        'datasets/cat2dog/testB',
        'datasets/cat2dog/testB_map',
    ]

    images_dirs = [
        'experiments/ours_horse2zebra_lr_sep_idt_foreonly2/generated/220000/A',
        'experiments/ours_horse2zebra_lr_sep_idt_foreonly2/generated/220000/M_A',
        'experiments/ours_horse2zebra_lr_sep_recons_idt/generated/220000/GA',
        'experiments/ours_horse2zebra_lr_sep_idt_foreonly2/generated/220000/GA',
    ]

    out_dir = os.path.join('temp', 'comp_h2z_foreonly')
    os.makedirs(out_dir, exist_ok=True)

    compare_images(images_dirs, out_dir)


