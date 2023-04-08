import argparse
import datetime
import importlib
import json
import os
import shutil
import time

import torch
import torch.utils
from torch import nn 
from torch.nn import functional as F
from easydict import EasyDict
from PIL import Image

from datasets.dataset import UnpairedImageDataset
from scripts.losses import GANLoss, cal_gradient_penalty
from scripts.utils import load_option, pad_tensor, send_line_notify, tensor2ndarray, arrange_images
from scripts.cal_fid import get_fid
from scripts.scheduler import CosineLRWarmup


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
        'total_step','lr_G','lr_D', 'loss_D_real', 'loss_D_fake', 'gp', 'loss_D', 'loss_G_fake', 'recons_loss', 'loss_G'
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
    
    loss_fn = GANLoss(gan_mode='wgangp').to(device)
    network_module_G = importlib.import_module(opt.network_module_G)
    netG = getattr(network_module_G, opt.model_type_G)(opt.netG).to(device)
    network_module_D = importlib.import_module(opt.network_module_D)
    netD = getattr(network_module_D, opt.model_type_D)(opt.netD).to(device)

    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    schedulerG = CosineLRWarmup(optimG, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)
    optimD = torch.optim.Adam(netD.parameters(), lr=opt.learning_rate_D, betas=opt.betas)
    schedulerD = CosineLRWarmup(optimD, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)

    if opt.pretrained_path:
        state_dict = torch.load(opt.pretrained_path, map_location=device)
        netG.load_state_dict(state_dict['netG_state_dict'], strict=True)
        optimG.load_state_dict(state_dict['optimG_state_dict'])

    train_dataset = UnpairedImageDataset(opt.trainA_path, opt.trainB_path, opt.input_resolution, opt.data_extention, opt.cache_images)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_dataset = UnpairedImageDataset(opt.testA_path, opt.testB_path, opt.input_resolution, opt.data_extention, opt.cache_images)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print('Start Training')
    start_time = time.time()
    total_step = 0 if opt.resume_step is None else opt.resume_step
    best_fid = float('inf')
    netG.train()
    for e in range(1, 42*opt.steps):
        for i, data in enumerate(train_loader):
            # [(B,C,H,W)]*F -> (B,F,C,H,W)
            A = data['A'].to(device)
            B = data['B'].to(device)

            # Training D
            netD.zero_grad()
            logits_real = netD(B)
            loss_D_real = opt.coef_adv*loss_fn(logits_real, target_is_real=True)

            fake = netG(A)
            logits_fake = netD(fake.detach())
            loss_D_fake = opt.coef_adv*loss_fn(logits_fake, target_is_real=False)

            gp = opt.coef_gp*cal_gradient_penalty(netD, B, fake, device)[0]

            loss_D = loss_D_real + loss_D_fake + gp
            loss_D.backward(retain_graph=True)
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netD.parameters(), opt.grad_clip_val)
            optimD.step()
            schedulerD.step()

            # Training G
            netG.zero_grad()
            logits_fake = netD(fake)
            loss_G_fake = opt.coef_adv*loss_fn(logits_fake, target_is_real=True)
            loss_recons = opt.coef_l1*F.mse_loss(A, fake)
            
            loss_G = loss_G_fake + loss_recons
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
                        fake = netG(A)
                    
                    A, B, fake = map(lambda x: tensor2ndarray(x)[0,:h,:w,:], [A, B, fake])

                    out_dir_A = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'A')
                    out_dir_B = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'B')
                    out_dir_fake = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'fake')

                    os.makedirs(out_dir_A, exist_ok=True)
                    os.makedirs(out_dir_B, exist_ok=True)
                    os.makedirs(out_dir_fake, exist_ok=True)
                    A, B, fake = map(lambda x: Image.fromarray(x), [A, B, fake])
                    A.save(os.path.join(out_dir_A, f'{i:04}.{opt.save_extention}'))
                    B.save(os.path.join(out_dir_B, f'{i:04}.{opt.save_extention}'))
                    fake.save(os.path.join(out_dir_fake, f'{i:04}.{opt.save_extention}'))
                    if opt.save_compare:
                        out_dir_compare = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'compare')
                        os.makedirs(out_dir_compare, exist_ok=True)
                        arrange_images([A, fake, B]).save(os.path.join(out_dir_compare, f'{i:04}.{opt.save_extention}'))
                netG.train()
                
                fid_score = get_fid([out_dir_fake, out_dir_B], batch_size=32, dims=2048, num_workers=2) 
                    
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
                        'optimG_state_dict': optimG.state_dict(),
                        'schedularG_state_dict': schedulerG.state_dict(),
                        'netD_state_dict': netD.state_dict(),
                        'optimD_state_dict': optimD.state_dict(),
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
                    'optimG_state_dict': optimG.state_dict(),
                    'schedularG_state_dict': schedulerG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimD_state_dict': optimD.state_dict(),
                    'schedularD_state_dict': schedulerD.state_dict(),
                }, os.path.join(model_ckpt_dir, f'{opt.name}_{str(total_step).zfill(len(str(opt.steps)))}.ckpt'))
                    
            if total_step==opt.steps:
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                    'schedularG_state_dict': schedulerG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimD_state_dict': optimD.state_dict(),
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
