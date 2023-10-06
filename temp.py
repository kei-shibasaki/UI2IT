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
    from scripts.utils import img2tensor

    img = cv2.imread('lena.png')
    img = cv2.resize(img, dsize=[256,256], interpolation=cv2.INTER_LINEAR)
    A = img2tensor(img, bgr2rgb=True).unsqueeze(0).to(device)

    grid_size = 2
    crop_size = 256
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

    #A = torch.rand((b,c,h,w)).to(device)

    print('-'*32)
    downsampled = F.interpolate(A, [64, 64], mode='bilinear', align_corners=True)
    source_control_points_A = loc(downsampled)
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
    net = U2NETP(out_ch=1)
    #net = ResnetGenerator(3,3,64,n_blocks=9)
    out = net(x)
    for o in out:
        print(o.shape)
    #torchinfo.summary(net, input_data=[x])

def test_dataset():
    from datasets.dataset import SingleImageDatasetWithMask, UnpairedImageDatasetWithMask

    source_path = 'datasets/horse2zebra/testA'
    source_mask_path = 'datasets/horse2zebra/testA_map/0'
    target_path = 'datasets/horse2zebra/testB'
    input_resolution = [256, 256]
    dataset = SingleImageDatasetWithMask(source_path, source_mask_path, input_resolution, cache_images=True)
    #dataset = UnpairedImageDatasetWithMask(source_path, source_mask_path, target_path, input_resolution, cache_images=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for data in dataloader:
        #A = data['A']
        #B = data['B']
        #mask_A = data['mask_A']
        #print(A.shape, B.shape)
        #temp = torch.cat([A, mask_A], dim=1)
        #print(temp.shape)
        #exit()

        img = data['img']
        mask = data['mask']
        print(img.shape, mask.shape)
        exit()

def test_net():
    from models.network import Generator
    from scripts.utils import load_option

    opt = EasyDict(load_option('config/config_mspc_one_sal.json'))

    b,c,h,w = 1,5,256,256
    x = torch.rand((b,c,h,w)).to(device)

    net = Generator(opt.netG).to(device)

    out = net(x)

    print(out.shape)
    
def select_images():
    generated_dir = 'experiments/TEST2/generated/220000'
    labels = ['A', 'GA', 'GA_f', 'GA_b', 'M_A', 'TM_GA']
    images = []
    for label in labels:
        if label=='GA':
            img_path = os.path.join(generated_dir, label, f'0011.png')
        else:
            img_path = os.path.join(generated_dir, label, f'0011.jpg')
        img = cv2.imread(img_path)
        images.append(img)
    
    for img, label in zip(images, labels):
        if label in ['GA_f', 'GA_b']:
            mask = cv2.cvtColor(images[5], cv2.COLOR_BGR2GRAY)/255.0
            if label=='GA_b':
                mask = 1.0 - mask
            mask = mask.reshape(256,256,1)
            print(mask.shape, img.shape)
            img = mask * img
        cv2.imwrite(f'temp2/{label}.jpg', img)

def test_compare_img():
    from scripts.utils import arrange_images

    images_dirs = [
        'experiments/MSPC_horse2zebra_b04/generated/198000/A',
        'experiments/MSPC_horse2zebra_b04/generated/198000/GA', 
        'experiments/TEST2/generated/220000/GA', 
    ]
    images_dirs = [
        'experiments/MSPC_paper_anime/generated/196000/A',
        'experiments/MSPC_paper_anime/generated/196000/GA', 
        'experiments/mspc_sal_mistm_anime/generated/690000/GA', 
    ]

    images_list = []
    for i, images_dir in enumerate(images_dirs):
        if i==0:
            images_list.append(sorted(glob.glob(os.path.join(images_dir, '*.jpg'))))
        else:
            images_list.append(sorted(glob.glob(os.path.join(images_dir, '*.png'))))

    n_images = len(images_list[0])
    n_row = len(images_list)
    #print(n_images, n_row)

    for i in tqdm(range(n_images)):
        images_path = [images_list[j][i] for j in range(n_row)]
        images = [Image.open(img_path).convert('RGB') for img_path in images_path]
        compare = arrange_images(images)
        compare.save(f'temp2/{i:03}.jpg')

def test_thres():
    out_dir  = os.path.join('temp', 'comp_thres')
    os.makedirs(out_dir, exist_ok=True)

    images_dirs = [
        'datasets/selfie2anime/testA',
        'datasets/selfie2anime/testA_map/0',
    ]

    images_list = []
    for i, images_dir in enumerate(images_dirs):
        if i==0:
            images_list.append(sorted(glob.glob(os.path.join(images_dir, '*.jpg'))))
        else:
            images_list.append(sorted(glob.glob(os.path.join(images_dir, '*.png'))))

    n_images = len(images_list[0])
    n_row = len(images_list)

    for i in tqdm(range(n_images)):
        A = cv2.imread(images_list[0][i])
        M = cv2.imread(images_list[1][i])
        M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)
        M_ths = []
        for th in [1, 5, 10, 15, 20, 25, 50, 100, 150, 200]:
            _, M_th = cv2.threshold(M, th, 255, cv2.THRESH_BINARY)
            M_ths.append(M_th)
        M = cv2.cvtColor(M, cv2.COLOR_GRAY2BGR)
        M_ths = [cv2.cvtColor(M_th, cv2.COLOR_GRAY2BGR) for M_th in M_ths]
        img = np.hstack([A,M,*M_ths])
        #print(img.shape)
        cv2.imwrite(os.path.join(out_dir, f'{i:03}.jpg'), img)

def check_params():
    from models.network import GeneratorSeparate, GeneratorSeparate2
    from models.resnet2 import ResnetGenerator
    import thop 

    from scripts.utils import load_option

    #opt = EasyDict(load_option('config/config_ours_anime3.json'))
    opt = EasyDict(load_option('config/config_ours.json'))

    b,c,h,w = 1,3,256,256

    #net = ResnetGenerator(4, 7, 64, n_blocks=9).to(device)
    net = GeneratorSeparate2(opt.netG).to(device)

    x = torch.rand((b,3,h,w)).to(device)
    mask = torch.rand((b,1,h,w)).to(device)

    fore, back, output = net(x, mask)
    print(fore.shape, back.shape, output.shape)

    macs, params = thop.profile(net, inputs=[x, mask])
    print(f'GMACs: {macs/1e9:.3f}, Params: {params/1e6:.3f} M')

def check_params_u2net():
    from models.u2net import U2NETP
    import thop 

    from scripts.utils import load_option

    b,c,h,w = 1,3,256,256

    #net = ResnetGenerator(4, 7, 64, n_blocks=9).to(device)
    net = U2NETP(in_ch=3, out_ch=1).to(device)

    x = torch.rand((b,3,h,w)).to(device)

    out, _, _, _, _, _, _ = net(x)

    macs, params = thop.profile(net, inputs=[x])
    print(f'GMACs: {macs/1e9:.3f}, Params: {params/1e6:.3f} M')

def count_epoch():
    ext = 'png'
    #dataset_name = 'edges2handbags'
    dataset_name = 'front2side'
    total_epoch = 50

    trainA_dir = f'datasets/{dataset_name}/trainA'
    trainB_dir = f'datasets/{dataset_name}/trainB'

    imagesA = glob.glob(os.path.join(trainA_dir, f'*.{ext}'))
    imagesB = glob.glob(os.path.join(trainB_dir, f'*.{ext}'))

    #step_per_epoch = min(len(imagesA), len(imagesB))
    step_per_epoch = max(len(imagesA), len(imagesB))
    print(f'Epoch: 1, Step: {step_per_epoch}')
    print(f'Epoch: {total_epoch}, Step: {step_per_epoch*total_epoch}')
    print(f'Epoch: {total_epoch}, Step: {step_per_epoch*total_epoch//4}')

def to_binary():
    out_dir = 'datasets/edges2handbags/trainB_map_bi'
    os.makedirs(out_dir, exist_ok=True)
    images = sorted(glob.glob('datasets/edges2handbags/trainB_map/*.png'))
    for img_path in tqdm(images):
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(out_dir, fname), thresh)

def check_binary():
    out_dir = 'temp/comp_thresh'
    os.makedirs(out_dir, exist_ok=True)
    images1 = sorted(glob.glob('datasets/edges2handbags/testB/*.jpg'))
    images2 = sorted(glob.glob('datasets/edges2handbags/testB_map/*.png'))
    for img_path1, img_path2 in zip(tqdm(images1), images2):
        fname = os.path.basename(img_path1)
        img_input = cv2.imread(img_path1)
        img = cv2.imread(img_path2)
        imgs = [img_input, img]
        for min_val in [1, 5, 10, 20, 30, 40, 50, 100, 200]:
            ret, thresh = cv2.threshold(img, min_val, 255, cv2.THRESH_BINARY)
            #thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            thresh = img_input * (thresh/255.0)
            imgs.append(thresh)

        img = np.hstack(imgs)
        cv2.imwrite(os.path.join(out_dir, fname), img)




if __name__=='__main__':
    #to_binary()
    #check_binary()
    count_epoch()