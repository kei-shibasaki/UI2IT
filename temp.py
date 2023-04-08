import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np
import itertools
from easydict import EasyDict
from tqdm import tqdm
import cv2 
from PIL import Image
import pandas as pd 
import glob 
import os 

device = torch.device('cuda')

def check_mspc():
    from models.mspc import UnBoundedGridLocNet, TPSGridGen, grid_sample, scale_constraint
    grid_size = 2
    crop_size = 128
    r1 = r2 = 0.9
    b,c,h,w = 1,3,256,256

    target_control_points = torch.Tensor(list(itertools.product(
        np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_size - 1)),
        np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_size - 1)),
    )))
    Y, X = target_control_points.split(1, dim=1)
    target_control_point = torch.cat([X, Y], dim=1).to(device)
    target_control_points = torch.cat([target_control_point.unsqueeze(0)]*b, dim=0)
    print(target_control_points.shape)

    loc = UnBoundedGridLocNet(grid_height=grid_size, grid_width=grid_size, target_control_point=target_control_points).to(device)
    tps = TPSGridGen().to(device)

    A = torch.rand((b,c,h,w)).to(device)
    B = torch.randn((b,c,h,w)).to(device)

    print('-'*32)
    source_control_points_A = loc(A)
    print(source_control_points_A.shape)
    source_coordinate_A = tps(source_control_points_A, target_control_points, crop_size, crop_size)
    print(source_coordinate_A.shape)
    grid_A = source_coordinate_A.view(b, crop_size, crop_size, 2)
    print(grid_A.shape)
    pert_A = grid_sample(A, grid_A)
    print(pert_A.shape)
    constraint_A = scale_constraint(source_control_points_A, target_control_points)
    print(constraint_A)
    cordinate_contraint_A = ((source_coordinate_A.mean(dim=1).abs()).clamp(min=0.25)).mean()
    print(cordinate_contraint_A)

    print('-'*32)
    source_control_points_B = loc(B)
    print(source_control_points_B.shape)
    source_coordinate_B = tps(source_control_points_B, target_control_points, crop_size, crop_size)
    print(source_coordinate_B.shape)
    grid_B = source_coordinate_B.view(b, crop_size, crop_size, 2)
    print(grid_B.shape)
    pert_B = grid_sample(B, grid_B)
    print(pert_B.shape)
    constraint_B = scale_constraint(source_control_points_B, target_control_points)
    print(constraint_B)
    cordinate_contraint_B = ((source_coordinate_B.mean(dim=1).abs()).clamp(min=0.25)).mean()
    print(cordinate_contraint_B)

    print('-'*32)
    loss_pert_constraint_D = (constraint_A+cordinate_contraint_A + constraint_B+cordinate_contraint_B)*0.5
    print(loss_pert_constraint_D)

def check_fid():
    from scripts.cal_fid import get_fid
    path1 = 'datasets/horse2zebra/testA'
    path2 = 'datasets/horse2zebra/testB'

    out = get_fid([path1, path2], batch_size=32, dims=2048, num_workers=2) 
    print(out)

def check_schedular():
    import matplotlib.pyplot as plt
    from models.lptn import LPTN
    from scripts.scheduler import CosineLRWarmup
    from scripts.utils import load_option

    opt = EasyDict(load_option('config/config_lptn.json'))

    netG = LPTN(opt.netG).to(device)
    optimG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=[0.9,0.999])
    schedularG = CosineLRWarmup(optimG, lr_w=5e-5, lr_max=1e-4, lr_min=1e-6, step_w=50000, step_max=300000)

    lrs = []
    for i in tqdm(range(300000)):
        lr = schedularG.get_lr()[0]
        lrs.append(lr)
        schedularG.step()
    
    plt.figure() 
    plt.plot(lrs)
    plt.savefig('temp.png')

def check_netP():
    from models.mspc import PerturbationNetwork
    from scripts.utils import load_option

    opt = EasyDict(load_option('config/config_mspc.json'))
    netP = PerturbationNetwork(**opt.netP).to(device)

    A = torch.rand((1,3,256,256)).to(device)
    B = torch.rand((1,3,256,256)).to(device)

    #grid_A, pert_A, constraint_A, cordinate_contraint_A, grid_B, pert_B, constraint_B, cordinate_contraint_B = netP(A, B)
    out = netP(A)

    for value in out:
        if isinstance(value, torch.Tensor):
            if value.shape!=torch.Size([]):
                print(f'{value.shape}')
            else:
                print(f'{value}')
        else:
            print(f'{value}')
    
def check_mspc2():
    from models.mspc import PerturbationNetwork, grid_sample
    from scripts.utils import read_img, arrange_images, tensor2ndarray

    state_dict_P = torch.load('experiments/TEST_MSPC/ckpt/TEST_MSPC_002000.ckpt', map_location=device)['netP_state_dict']

    grid_size = 2
    crop_size = 256
    r1 = r2 = 0.9
    pert_threshold = 2.0
    b,c,h,w = 1,3,256,256

    A = read_img('datasets/horse2zebra/trainA/n02381460_2.jpg').unsqueeze(0)
    A = F.interpolate(A, size=[256, 256], mode='bilinear', align_corners=False)
    A = A.to(device)

    B = read_img('datasets/horse2zebra/trainB/n02391049_2.jpg').unsqueeze(0)
    B = F.interpolate(B, size=[256, 256], mode='bilinear', align_corners=False)
    B = B.to(device)

    netP = PerturbationNetwork(grid_size, crop_size, r1, r2, pert_threshold, device).to(device)
    netP.load_state_dict(state_dict_P, strict=True)

    grid_A, pert_A, constraint_A, cordinate_contraint_A = netP(A)
    grid_B, pert_B, constraint_B, cordinate_contraint_B = netP(B)

    A, pert_A, B, pert_B = map(lambda x: tensor2ndarray(x)[0,:,:,:].astype(np.uint8), [A, pert_A, B, pert_B])

    A, pert_A, B, pert_B = map(lambda x: Image.fromarray(x), [A, pert_A, B, pert_B])

    compare = arrange_images([A, pert_A, B, pert_B])
    compare.save('temp.png')

def check_csv():
    df = pd.read_csv('temp2.csv', index_col=None)

    print(df.head())

def rewrite_csv():
    with open('temp.csv', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    with open('temp2.csv', 'w', encoding='utf-8') as fp:
        for i, line in enumerate(tqdm(lines)):
            if i!=0:
                fp.write(f'{line[:-2]}\n')
            else:
                fp.write(line)

def test_u2net():
    from models.u2net import U2NETP, U2NET
    from models.resnet import ResnetGenerator
    import torchinfo

    b,c,h,w = 1,3,256,256
    x = torch.rand((b,c,h,w))
    net = U2NETP(out_ch=3)
    #net = ResnetGenerator(3,3,64,n_blocks=9)
    #out = net(x)
    torchinfo.summary(net, input_data=[x])
    
def test_compare_img():
    from scripts.utils import arrange_images
    images_dirs = [
        'experiments/TEST_cycle_horse2zebra/generated/001000/A',
        'experiments/TEST_cycle_horse2zebra/generated/100000/GA', 
        'experiments/TEST_cycle_horse2zebra_ema/generated/100000/GA', 
    ]
    images_list = [sorted(glob.glob(os.path.join(d, '*.jpg'))) for d in images_dirs]
    n_images = len(images_list[0])
    n_row = len(images_list)
    #print(n_images, n_row)

    for i in tqdm(range(n_images)):
        images_path = [images_list[j][i] for j in range(n_row)]
        images = [Image.open(img_path).convert('RGB') for img_path in images_path]
        compare = arrange_images(images)
        compare.save(f'temp/{i:03}.jpg')

if __name__=='__main__':
    test_compare_img()