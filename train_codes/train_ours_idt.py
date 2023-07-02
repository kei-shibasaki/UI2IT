import argparse
import datetime
import importlib
import json
import os
import shutil
import time
import itertools

import numpy as np 
import torch
import torch.utils
from torch import nn 
from torch.nn import functional as F
from easydict import EasyDict
from PIL import Image

from models.mspc import PerturbationNetwork, grid_sample
from datasets.dataset import UnpairedImageDatasetWithMask, SingleImageDatasetWithMask
from scripts.losses import GANLoss
from scripts.utils import load_option, pad_tensor, send_line_notify, tensor2ndarray, arrange_images
from scripts.training_utils import set_requires_grad
from scripts.cal_fid import get_fid
from scripts.scheduler import LinearLRWarmup


def train(opt_path):
    opt = EasyDict(load_option(opt_path))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    model_ckpt_dir = f'./experiments/{opt.name}/ckpt'
    image_out_dir = f'./experiments/{opt.name}/generated'
    log_dir = f'./experiments/{opt.name}/logs'
    log_path = f'{log_dir}/log_{opt.name}.log'
    log_train_losses_path = f'{log_dir}/train_losses_{opt.name}.csv'
    log_test_losses_paths = f'{log_dir}/test_losses_{opt.name}.csv'

    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_items_train = [
        'total_step','lr', 'loss_G_GA', 'loss_G_GTA', 'gta_tga_distance_G', 'loss_G_background_recons', 'loss_G_idt', 'loss_G_idt_pert', 'loss_G',
        'loss_D_B', 'loss_D_GA', 'loss_D_TB', 'loss_D_GTA', 'gta_tga_distance_D', 'loss_pert_constraint_D', 'loss_D'
    ]
    log_items_val = [
        'total_step', 'fid_score'
    ]

    with open(log_path, mode='w', encoding='utf-8') as fp: fp.write('')
    with open(log_train_losses_path, mode='w', encoding='utf-8') as fp: 
        fp.write(','.join(log_items_train)+'\n')
    with open(log_test_losses_paths, mode='w', encoding='utf-8') as fp:
        fp.write(','.join(log_items_val)+'\n')
    
    shutil.copy(opt_path, f'./experiments/{opt.name}/{os.path.basename(opt_path)}')
    
    loss_fn = GANLoss(gan_mode='lsgan').to(device)
    network_module_G = importlib.import_module(opt.network_module_G)
    network_module_D = importlib.import_module(opt.network_module_D)

    netG = getattr(network_module_G, opt.model_type_G)(opt.netG).to(device)
    netD = getattr(network_module_D, opt.model_type_D)(opt.netD).to(device)
    netP = PerturbationNetwork(**opt.netP).to(device)

    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate, betas=opt.betas)
    schedulerG = LinearLRWarmup(optimG, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)
    optimD = torch.optim.Adam(netD.parameters(), lr=opt.learning_rate, betas=opt.betas)
    schedulerD = LinearLRWarmup(optimD, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)
    optimP = torch.optim.Adam(netP.parameters(), lr=opt.learning_rate, betas=opt.betas)
    schedulerP = LinearLRWarmup(optimP, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)


    if opt.pretrained_path:
        state_dict = torch.load(opt.pretrained_path, map_location=device)
        netG.load_state_dict(state_dict['netG_state_dict'], strict=True)
        netP.load_state_dict(state_dict['netP_state_dict'], strict=True)
        netD.load_state_dict(state_dict['netD_state_dict'], strict=True)
        optimG.load_state_dict(state_dict['optimG_state_dict'])
        optimP.load_state_dict(state_dict['optimP_state_dict'])
        optimD.load_state_dict(state_dict['optimD_state_dict'])
        schedulerG.load_state_dict(state_dict['schedularG_state_dict'])
        schedulerP.load_state_dict(state_dict['schedularP_state_dict'])
        schedulerD.load_state_dict(state_dict['schedularD_state_dict'])

    train_dataset = UnpairedImageDatasetWithMask(opt.trainA_path, opt.trainA_mask_path, opt.trainB_path, opt.trainB_mask_path, opt.input_resolution, opt.data_extention, opt.mask_extention, opt.cache_images)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_dataset = SingleImageDatasetWithMask(opt.testA_path, opt.testA_mask_path, opt.input_resolution, opt.data_extention, opt.mask_extention, opt.cache_images)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print('Start Training')
    start_time = time.time()
    total_step = 0 if opt.resume_step is None else opt.resume_step
    best_fid = float('inf')
    netD.train()
    netP.train()
    netG.train()

    for e in range(1, 42*opt.steps):
        for i, data in enumerate(train_loader):
            A = data['A'].to(device)
            B = data['B'].to(device)
            M_A = data['mask_A'].to(device)
            M_B = data['mask_B'].to(device)
            # G(A)
            GA_f, GA_b, GA = netG(A, M_A)
            # T(A)
            grid_A, TA, constraint_A, cordinate_contraint_A = netP(A)
            M_TA = grid_sample(M_A, grid_A.detach())
            # G(T(A))
            GTA_f, GTA_b, GTA = netG(TA.detach(), M_TA.detach())
            # T(G(A))
            TGA = grid_sample(GA, grid_A.detach())
            # G(B)
            GB_f, GB_b, GB = netG(B, M_B)
            # T(B)
            grid_B, TB, constraint_B, cordinate_contraint_B = netP(B)
            M_TB = grid_sample(M_B, grid_B.detach())
            # G(T(B))
            GTB_f, GTB_b, GTB = netG(TB.detach(), M_TB.detach())

            # Training G
            set_requires_grad([netD, netP], False)
            netG.zero_grad()
            logits_GA = netD(GA)
            logits_GTA = netD(GTA)
            loss_G_GA = opt.coef_adv*loss_fn(logits_GA, target_is_real=True)
            loss_G_GTA = opt.coef_adv*loss_fn(logits_GTA, target_is_real=True)
            gta_tga_distance_G = opt.coef_mspc*F.l1_loss(GTA, TGA)
            loss_G_idt = opt.coef_idt*F.l1_loss(B, GB)
            loss_G_idt_pert = opt.coef_idt*F.l1_loss(TB.detach(), GTB)
            loss_G_background_recons = opt.coef_recons*F.l1_loss(GA_b*(1-M_A), A*(1-M_A))
            
            loss_G = loss_G_GA + loss_G_GTA + gta_tga_distance_G  + loss_G_background_recons + loss_G_idt + loss_G_idt_pert
            loss_G.backward()
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netG.parameters(), opt.grad_clip_val)
            optimG.step()
            schedulerG.step()

            # Training D
            set_requires_grad([netD, netP], True)
            netP.zero_grad()
            netD.zero_grad()

            GTA_f, GTA_b, GTA = netG(TA, M_TA)
            TGA = grid_sample(GA.detach(), grid_A)
            logits_B = netD(B)
            logits_GA = netD(GA.detach())
            logits_TB = netD(TB)
            logits_GTA = netD(GTA)

            gta_tga_distance_D = opt.coef_mspc*F.l1_loss(GTA.detach(), TGA)

            loss_D_B = opt.coef_adv*loss_fn(logits_B, target_is_real=True)
            loss_D_GA = opt.coef_adv*loss_fn(logits_GA, target_is_real=False)
            loss_D_TB = opt.coef_adv*loss_fn(logits_TB, target_is_real=True)
            loss_D_GTA = opt.coef_adv*loss_fn(logits_GTA, target_is_real=False)

            loss_pert_constraint_D = -opt.coef_constraint*(constraint_A + cordinate_contraint_A + constraint_B + cordinate_contraint_B)
            
            loss_D = loss_D_B + loss_D_GA + loss_D_TB + loss_D_GTA + gta_tga_distance_D + loss_pert_constraint_D

            loss_D.backward()
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netD.parameters(), opt.grad_clip_val)
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netP.parameters(), opt.grad_clip_val)
            optimD.step()
            optimP.step()
            schedulerD.step()
            
            total_step += 1

            if total_step%1==0:
                lr = [group['lr'] for group in optimG.param_groups][0]
                lg = ''
                for logname_idx, name in enumerate(log_items_train):
                    val = eval(name)
                    if logname_idx!=len(log_items_train)-1:
                        lg = lg + f'{val:f},'
                    else:
                        lg = lg + f'{val:f}'
                with open(log_train_losses_path, mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')
            
            if total_step%opt.print_freq==0 or total_step==1:
                rest_step = opt.steps-total_step
                time_per_step = int(time.time()-start_time) / total_step if opt.resume_step is None else int(time.time()-start_time) / (total_step-opt.resume_step)
                elapsed = datetime.timedelta(seconds=int(time.time()-start_time))
                eta = datetime.timedelta(seconds=int(rest_step*time_per_step))
                lg = f'{total_step}/{opt.steps}, Epoch:{str(e).zfill(len(str(opt.steps)))}, elepsed: {elapsed}, eta: {eta}, '
                for logname_idx, name in enumerate(log_items_train):
                    val = eval(name)
                    if logname_idx!=len(log_items_train)-1:
                        lg = lg + f'{name}: {val:f}, '
                    else:
                        lg = lg + f'{name}: {val:f}'
                print(lg)
                with open(log_path, mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')

            if total_step%opt.eval_freq==0 or total_step==opt.steps:
                # Validation
                netG.eval()
                netP.eval()
                for i, data in enumerate(val_loader):
                    A = data['img'].to(device)
                    M_A = data['mask'].to(device)
                    b,c,h,w = A.shape
                    A = pad_tensor(A, divisible_by=2**3)
                    M_A = pad_tensor(M_A, divisible_by=2**3)
                    with torch.no_grad():
                        GA_f, GA_b, GA = netG(A, M_A)
                        grid_A, TA, constraint_A, cordinate_contraint_A = netP(A)
                        M_TA = grid_sample(M_A, grid_A.detach())
                        GTA_f, GTA_b, GTA = netG(TA, M_TA)
                        TGA = grid_sample(GA, grid_A)

                    images_save = [A, M_A, GA_f, GA_b, GA, TA, M_TA, GTA_f, GTA_b, GTA, TGA]
                    labels = ['A', 'M_A', 'GA_f', 'GA_b', 'GA', 'TA', 'M_TA', 'GTA_f', 'GTA_b', 'GTA', 'TGA']
                    out_dirs = [os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', f'{label}') for label in labels]
                    out_dir_GA = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', f'GA')
                    for out_dir in out_dirs: os.makedirs(out_dir, exist_ok=True)
                    images_save = list(map(lambda x: tensor2ndarray(x)[0,:h,:w,:], images_save))
                    for img_idx in range(len(images_save)):
                        if images_save[img_idx].shape[2]==1:
                            images_save[img_idx] = np.squeeze(images_save[img_idx], -1)
                            
                    images_save = list(map(lambda x: Image.fromarray(x), images_save))
                    for img, label, out_dir in zip(images_save, labels, out_dirs):
                        if label=='GA':
                            img.save(os.path.join(out_dir, f'{i:04}.png'))
                        else:
                            img.save(os.path.join(out_dir, f'{i:04}.{opt.save_extention}'))

                    if opt.save_compare:
                        out_dir_compare = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'compare')
                        os.makedirs(out_dir_compare, exist_ok=True)
                        arrange_images(images_save).save(os.path.join(out_dir_compare, f'{i:04}.{opt.save_extention}'))
                netG.train()
                netP.train()
                
                fid_score = get_fid([out_dir_GA, opt.testB_path], batch_size=32, dims=2048, num_workers=2) 
                    
                txt = f'FID: {fid_score:f}'
                # print(txt)
                with open(log_path, mode='a', encoding='utf-8') as fp:
                    fp.write(txt+'\n')
                lg_val = ''
                for logname_idx, name in enumerate(log_items_val):
                    val = eval(name)
                    if logname_idx!=len(log_items_val)-1:
                        lg_val = lg_val + f'{val:f},'
                    else:
                        lg_val = lg_val + f'{val:f}'
                with open(log_test_losses_paths, mode='a', encoding='utf-8') as fp:
                    fp.write(lg_val+'\n')
                
                if fid_score <= best_fid:
                    best_fid = fid_score
                    torch.save({
                        'total_step': total_step,
                        'netG_state_dict': netG.state_dict(),
                        'netP_state_dict': netP.state_dict(),
                        'netD_state_dict': netD.state_dict(),
                        'optimG_state_dict': optimG.state_dict(),
                        'optimP_state_dict': optimP.state_dict(),
                        'optimD_state_dict': optimD.state_dict(),
                        'schedularG_state_dict': schedulerG.state_dict(),
                        'schedularP_state_dict': schedulerP.state_dict(),
                        'schedularD_state_dict': schedulerD.state_dict(),
                    }, os.path.join(model_ckpt_dir, f'{opt.name}_best.ckpt'))
                
                if total_step%opt.nortify_freq==0 and opt.enable_line_nortify:
                    with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                        token = json.load(fp)['token']
                    send_line_notify(token, f'{opt.name} Step: {total_step}\n{lg}\n{lg_val}\n{txt}')
            
            if total_step%opt.save_freq==0:
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'netP_state_dict': netP.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                    'optimP_state_dict': optimP.state_dict(),
                    'optimD_state_dict': optimD.state_dict(),
                    'schedularG_state_dict': schedulerG.state_dict(),
                    'schedularP_state_dict': schedulerP.state_dict(),
                    'schedularD_state_dict': schedulerD.state_dict(),
                }, os.path.join(model_ckpt_dir, f'{opt.name}_{str(total_step).zfill(len(str(opt.steps)))}.ckpt'))
                    
            if total_step==opt.steps:
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'netP_state_dict': netP.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                    'optimP_state_dict': optimP.state_dict(),
                    'optimD_state_dict': optimD.state_dict(),
                    'schedularG_state_dict': schedulerG.state_dict(),
                    'schedularP_state_dict': schedulerP.state_dict(),
                    'schedularD_state_dict': schedulerD.state_dict(),
                }, os.path.join(model_ckpt_dir, f'{opt.name}_{str(total_step).zfill(len(str(opt.steps)))}.ckpt'))

                if opt.enable_line_nortify:
                    with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                        token = json.load(fp)['token']
                    send_line_notify(token, f'Complete training {opt.name}.')
                
                print('Completed.')
                exit()
                

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of training network with adversarial loss.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    
    train(args.config)
