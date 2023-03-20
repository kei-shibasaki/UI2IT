import argparse
import datetime
import importlib
import json
import os
import shutil
import time
import itertools

import torch
import torch.utils
from torch import nn 
from torch.nn import functional as F
from easydict import EasyDict
from PIL import Image

from models.mspc import PerturbationNetwork, grid_sample
from datasets.dataset import UnpairedImageDataset
from scripts.losses import GANLoss, cal_gradient_penalty
from scripts.utils import load_option, pad_tensor, send_line_notify, tensor2ndarray, arrange_images, set_requires_grad
from scripts.cal_fid import get_fid
from scripts.optimizer import CosineLRWarmup


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
        'total_step','lr_G','lr_D', 'loss_D_B', 'loss_D_GA', 'loss_D_TB', 'loss_D_GTA', 'loss_D_perturbation', 'loss_D', 
        'loss_G_GA', 'loss_G_GTA', 'gta_tga_distance', 'loss_G'
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
    netG = getattr(network_module_G, opt.model_type_G)(**opt.netG).to(device)
    network_module_D = importlib.import_module(opt.network_module_D)
    netD = getattr(network_module_D, opt.model_type_D)(opt.netD).to(device)
    netP = PerturbationNetwork(**opt.netP).to(device)
    netD_perturbation = getattr(network_module_D, opt.model_type_D)(opt.netD).to(device)

    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    schedulerG = CosineLRWarmup(optimG, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)
    optimD = torch.optim.Adam(itertools.chain(netD.parameters(), netD_perturbation.parameters()), lr=opt.learning_rate_D, betas=opt.betas)
    schedulerD = CosineLRWarmup(optimD, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)
    optimP = torch.optim.Adam(netP.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    schedulerP = CosineLRWarmup(optimP, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)

    train_dataset = UnpairedImageDataset(opt.trainA_path, opt.trainB_path, opt.input_resolution, opt.data_extention, opt.cache_images)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_dataset = UnpairedImageDataset(opt.testA_path, opt.testB_path, opt.input_resolution, opt.data_extention, opt.cache_images)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print('Start Training')
    start_time = time.time()
    total_step = 0 if opt.resume_step is None else opt.resume_step
    best_fid = float('inf')
    netD.train()
    netD_perturbation.train()
    netP.train()
    netG.train()

    for e in range(1, 42*opt.steps):
        for i, data in enumerate(train_loader):
            # [(B,C,H,W)]*F -> (B,F,C,H,W)
            A = data['A'].to(device)
            B = data['B'].to(device)

            set_requires_grad([netD, netD_perturbation, netP], True)

            # G(A)
            GA = netG(A)
            
            set_requires_grad([netG], False)

            # T(A)
            grid_A, TA, constraint_A, cordinate_contraint_A = netP(A)
            # T(B)
            grid_B, TB, constraint_B, cordinate_contraint_B = netP(B)

            # G(T(A))
            GTA = netG(TA)
            # T(G(A))
            TGA = grid_sample(GA, grid_A)
            gta_tga_distance = opt.coef_mspc*F.l1_loss(GTA, TGA)

            loss_pert_constraint_D = constraint_A + cordinate_contraint_A + constraint_B + cordinate_contraint_B

            # Training D
            netP.zero_grad()
            netD.zero_grad()
            netD_perturbation.zero_grad()

            logits_B = netD(B)
            loss_D_B = loss_fn(logits_B, target_is_real=True)
            logits_GA = netD(GA.detach())
            loss_D_GA = loss_fn(logits_GA, target_is_real=False)

            logits_TB = netD_perturbation(TB)
            loss_D_TB = loss_fn(logits_TB, target_is_real=True)
            logits_GTA = netD_perturbation(GTA.detach())
            loss_D_GTA = loss_fn(logits_GTA, target_is_real=False)

            loss_D_perturbation = gta_tga_distance - opt.coef_constraint*loss_pert_constraint_D

            # gp_D = opt.coef_gp*cal_gradient_penalty(netD, B, GA.detach(), device)[0]
            # gp_D_perturbation = opt.coef_gp*cal_gradient_penalty(netD_perturbation, TB, GTA.detach(), device)[0]
            
            loss_D = loss_D_B + loss_D_GA + loss_D_TB + loss_D_GTA + loss_D_perturbation

            loss_D.backward(retain_graph=True)
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netD.parameters(), opt.grad_clip_val)
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netD_perturbation.parameters(), opt.grad_clip_val)
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netP.parameters(), opt.grad_clip_val)
            optimD.step()
            optimP.step()
            schedulerD.step()
            schedulerP.step()

            # Training G
            netG.zero_grad()
            set_requires_grad([netD, netD_perturbation, netP], False)
            set_requires_grad([netG], True)
            # netPが更新されるからTA, GTA, TGAを生成し直すほうが良いかも
            grid_A, TA, constraint_A, cordinate_contraint_A = netP(A)
            GTA = netG(TA)
            TGA = grid_sample(GA, grid_A)
            gta_tga_distance = opt.coef_mspc*F.l1_loss(GTA, TGA)

            logits_GA_forG = netD(GA)
            loss_G_GA = loss_fn(logits_GA_forG, target_is_real=True)
            logits_GTA_forG = netD_perturbation(GTA)
            loss_G_GTA = loss_fn(logits_GTA_forG, target_is_real=True)
            
            loss_G = loss_G_GA + loss_G_GTA + gta_tga_distance
            loss_G.backward()
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netG.parameters(), opt.grad_clip_val)
            optimG.step()
            schedulerG.step()
            
            total_step += 1

            if total_step%1==0:
                lr_G = [group['lr'] for group in optimG.param_groups][0]
                lr_D = [group['lr'] for group in optimD.param_groups][0]
                lg = ''
                for logname_idx, name in enumerate(log_items_train):
                    val = eval(name)
                    if logname_idx!=len(log_items_train):
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
                    if logname_idx!=len(log_items_train):
                        lg = lg + f'{name}: {val:f}, '
                    else:
                        lg = lg + f'{name}: {val:f}'
                print(lg)
                with open(log_path, mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')

            if total_step%opt.eval_freq==0:
                # Validation
                netG.eval()
                for i, data in enumerate(val_loader):
                    # [(B,C,H,W)]*F -> (B,F,C,H,W)
                    A = data['A'].to(device)
                    B = data['B'].to(device)
                    b,c,h,w = A.shape
                    A, B = map(lambda x: pad_tensor(x, divisible_by=2**3), [A, B])
                    with torch.no_grad():
                        GA = netG(A)
                    
                    A, B, GA = map(lambda x: tensor2ndarray(x)[0,:h,:w,:], [A, B, GA])

                    out_dir_A = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'A')
                    out_dir_B = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'B')
                    out_dir_GA = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'GA')

                    os.makedirs(out_dir_A, exist_ok=True)
                    os.makedirs(out_dir_B, exist_ok=True)
                    os.makedirs(out_dir_GA, exist_ok=True)
                    A, B, GA = map(lambda x: Image.fromarray(x), [A, B, GA])
                    A.save(os.path.join(out_dir_A, f'{i:04}.{opt.save_extention}'))
                    B.save(os.path.join(out_dir_B, f'{i:04}.{opt.save_extention}'))
                    GA.save(os.path.join(out_dir_GA, f'{i:04}.{opt.save_extention}'))
                    if opt.save_compare:
                        out_dir_compare = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'compare')
                        os.makedirs(out_dir_compare, exist_ok=True)
                        arrange_images([A, GA, B]).save(os.path.join(out_dir_compare, f'{i:04}.{opt.save_extention}'))
                netG.train()
                
                fid_score = get_fid([out_dir_GA, out_dir_B], batch_size=32, dims=2048, num_workers=2) 
                    
                txt = f'FID: {fid_score:f}'
                # print(txt)
                with open(log_path, mode='a', encoding='utf-8') as fp:
                    fp.write(txt+'\n')
                lg = ''
                for logname_idx, name in enumerate(log_items_val):
                    val = eval(name)
                    if logname_idx!=len(log_items_val):
                        lg = lg + f'{val:f},'
                    else:
                        lg = lg + f'{val:f}'
                with open(log_test_losses_paths, mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')
                
                if fid_score <= best_fid:
                    best_fid = fid_score
                    torch.save({
                        'total_step': total_step,
                        'netG_state_dict': netG.state_dict(),
                        'netP_state_dict': netP.state_dict(),
                        'netD_state_dict': netD.state_dict(),
                        'netD_perturbation': netD_perturbation.state_dict(), 
                        'optimG_state_dict': optimG.state_dict(),
                        'optimP_state_dict': optimP.state_dict(),
                        'optimD_state_dict': optimD.state_dict(),
                        'schedularG_state_dict': schedulerG.state_dict(),
                        'schedularP_state_dict': schedulerP.state_dict(),
                        'schedularD_state_dict': schedulerD.state_dict(),
                    }, os.path.join(model_ckpt_dir, f'{opt.name}_best.ckpt'))
                
                if total_step%opt.save_freq==0 and opt.enable_line_nortify:
                    with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                        token = json.load(fp)['token']
                    send_line_notify(token, f'{opt.name} Step: {total_step}\n{lg}\n{txt}')
            
            if total_step%opt.save_freq==0:
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'netP_state_dict': netP.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'netD_perturbation': netD_perturbation.state_dict(), 
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
                    'netD_perturbation': netD_perturbation.state_dict(), 
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
