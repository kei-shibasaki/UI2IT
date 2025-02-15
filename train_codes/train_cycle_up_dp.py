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

from datasets.dataset import UnpairedImageDataset, SimgleImageDataset
from scripts.losses import GANLoss, calc_r1_loss
from scripts.utils import load_option, pad_tensor, send_line_notify, tensor2ndarray, arrange_images
from scripts.training_utils import set_requires_grad, ImagePool, EMA
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
        'total_step', 'lr', 'loss_G_GA', 'loss_G_GB', 'loss_G_cycle_A', 'loss_G_cycle_B', 'loss_idt_A', 'loss_idt_B', 'loss_G',
        'loss_D_B', 'loss_D_GA', 'loss_D_AB', 'loss_D_BA_A', 'loss_D_BA_GB', 'loss_D_BA', 'loss_D'
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
    network_module_G_BA = importlib.import_module(opt.network_module_G_BA)
    network_module_D = importlib.import_module(opt.network_module_D)
    network_module_D_BA = importlib.import_module(opt.network_module_D_BA)

    netG = getattr(network_module_G, opt.model_type_G)(**opt.netG).to(device)
    netG_previous = getattr(network_module_G, opt.model_type_G)(**opt.netG).to(device)
    netG_BA = getattr(network_module_G_BA, opt.model_type_G_BA)(**opt.netG_BA).to(device)
    netG_BA_previous = getattr(network_module_G_BA, opt.model_type_G_BA)(**opt.netG_BA).to(device)
    netD = getattr(network_module_D, opt.model_type_D)(opt.netD).to(device)
    netD_BA = getattr(network_module_D_BA, opt.model_type_D_BA)(opt.netD_BA).to(device)

    netG, netG_previous, netG_BA, netG_BA_previous, netD, netD_BA = map(lambda x: nn.DataParallel(x, device_ids=[0,1,2,3]), [netG, netG_previous, netG_BA, netG_BA_previous, netD, netD_BA])

    GA_pool = ImagePool(opt.pool_size)
    GB_pool = ImagePool(opt.pool_size)

    EMA_G = EMA(netG, netG_previous, decay=opt.G_ema_decay)
    EMA_G_BA = EMA(netG_BA, netG_BA_previous, decay=opt.G_ema_decay)

    optimG = torch.optim.AdamW(itertools.chain(netG.parameters(), netG_BA.parameters()), lr=opt.learning_rate, betas=opt.betas, weight_decay=opt.adamw_decay)
    schedulerG = LinearLRWarmup(optimG, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)
    optimD = torch.optim.Adam(itertools.chain(netD.parameters(), netD_BA.parameters()), lr=opt.learning_rate, betas=opt.betas, weight_decay=opt.adamw_decay)
    schedulerD = LinearLRWarmup(optimD, opt.lr_w, opt.lr_max, opt.lr_min, opt.step_w, opt.step_max)

    train_dataset = UnpairedImageDataset(opt.trainA_path, opt.trainB_path, opt.input_resolution, opt.data_extention, opt.cache_images)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_dataset = SimgleImageDataset(opt.testA_path, opt.input_resolution, opt.data_extention, opt.cache_images)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print('Start Training')
    start_time = time.time()
    total_step = 0 if opt.resume_step is None else opt.resume_step
    best_fid = float('inf')
    netG.train()
    for e in range(1, 42*opt.steps):
        for i, data in enumerate(train_loader):
            A = data['A'].to(device)
            B = data['B'].to(device)
            GA = netG(A)
            GB = netG_BA(B)

            # Training D
            set_requires_grad([netD, netD_BA], True)
            netD.zero_grad()
            logits_B = netD(B)
            loss_D_B = opt.coef_adv*loss_fn(logits_B, target_is_real=True)
            logits_GA = netD(GA.detach())
            loss_D_GA = opt.coef_adv*loss_fn(logits_GA, target_is_real=False)
            loss_D_AB = loss_D_B + loss_D_GA

            logits_D_BA_A = netD_BA(A)
            loss_D_BA_A = opt.coef_adv*loss_fn(logits_D_BA_A, target_is_real=True)
            logits_D_BA_GB = netD_BA(GB.detach())
            loss_D_BA_GB = opt.coef_adv*loss_fn(logits_D_BA_GB, target_is_real=False)
            loss_D_BA = loss_D_BA_A + loss_D_BA_GB

            """
            if total_step%opt.G_ema_interval==0:
                loss_D_R1 = opt.coef_r1*calc_r1_loss(netD, B)
                loss_D_R1_BA = opt.coef_r1*calc_r1_loss(netD_BA, A)
            else:
                loss_D_R1 = 0.0
                loss_D_R1_BA = 0.0
            """

            loss_D = loss_D_AB + loss_D_BA #+ loss_D_R1 + loss_D_R1_BA
            loss_D.backward()

            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netD.parameters(), opt.grad_clip_val)
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netD_BA.parameters(), opt.grad_clip_val)
            optimD.step()
            schedulerD.step()

            # Training G
            set_requires_grad([netD, netD_BA], False)
            netG.zero_grad()
            A_rec = netG_BA(GA)
            B_rec = netG(GB)
            A_idt = netG_BA(A)
            B_idt = netG(B)
            logits_GA = netD(GA)
            loss_G_GA = opt.coef_adv*loss_fn(logits_GA, target_is_real=True)
            logits_GB = netD_BA(GB)
            loss_G_GB = opt.coef_adv*loss_fn(logits_GB, target_is_real=True)
            loss_G_cycle_A = opt.coef_cycle*F.l1_loss(A, A_rec)
            loss_G_cycle_B = opt.coef_cycle*F.l1_loss(B, B_rec)
            loss_idt_A = opt.coef_idt*F.l1_loss(A, A_idt)
            loss_idt_B = opt.coef_idt*F.l1_loss(B, B_idt)

            loss_G = loss_G_GA + loss_G_GB + loss_G_cycle_A + loss_G_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netG.parameters(), opt.grad_clip_val)
            if opt.use_grad_clip: torch.nn.utils.clip_grad_norm_(netG_BA.parameters(), opt.grad_clip_val)
            optimG.step()
            schedulerG.step()

            EMA_G.update(total_step)
            EMA_G_BA.update(total_step)
            
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

            if total_step%opt.eval_freq==0:
                # Validation
                netG.eval()
                netG_BA.eval()
                for i, A in enumerate(val_loader):
                    # [(B,C,H,W)]*F -> (B,F,C,H,W)
                    A = A.to(device)
                    b,c,h,w = A.shape
                    A = pad_tensor(A, divisible_by=2**3)
                    with torch.no_grad():
                        GA = netG(A)
                        A_rec = netG_BA(GA)
                    
                    A, GA, A_rec = map(lambda x: tensor2ndarray(x)[0,:h,:w,:], [A, GA, A_rec])

                    out_dir_A = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'A')
                    out_dir_GA = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'GA')
                    out_dir_A_rec = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'A_rec')

                    os.makedirs(out_dir_A, exist_ok=True)
                    os.makedirs(out_dir_GA, exist_ok=True)
                    os.makedirs(out_dir_A_rec, exist_ok=True)
                    A, GA, A_rec = map(lambda x: Image.fromarray(x), [A, GA, A_rec])
                    A.save(os.path.join(out_dir_A, f'{i:04}.{opt.save_extention}'))
                    GA.save(os.path.join(out_dir_GA, f'{i:04}.{opt.save_extention}'))
                    A_rec.save(os.path.join(out_dir_A_rec, f'{i:04}.{opt.save_extention}'))
                    if opt.save_compare:
                        out_dir_compare = os.path.join(image_out_dir, f'{str(total_step).zfill(len(str(opt.steps)))}', 'compare')
                        os.makedirs(out_dir_compare, exist_ok=True)
                        arrange_images([A, GA, A_rec]).save(os.path.join(out_dir_compare, f'{i:04}.{opt.save_extention}'))
                netG.train()
                
                fid_score = get_fid([out_dir_GA, out_dir_A], batch_size=32, dims=2048, num_workers=2) 

                    
                txt = f'FID: {fid_score:f}'
                # print(txt)
                with open(log_path, mode='a', encoding='utf-8') as fp:
                    fp.write(txt+'\n')
                lg = ''
                for logname_idx, name in enumerate(log_items_val):
                    val = eval(name)
                    if logname_idx!=len(log_items_val)-1:
                        lg = lg + f'{val:f},'
                    else:
                        lg = lg + f'{val:f}'
                with open(log_test_losses_paths, mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')
                
                if fid_score <= best_fid:
                    best_fid = fid_score
                    torch.save({
                        'total_step': total_step,
                        'netG_state_dict': netG.module.state_dict(),
                        'optimG_state_dict': optimG.state_dict(),
                        'schedularG_state_dict': schedulerG.state_dict(),
                        'netD_state_dict': netD.module.state_dict(),
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
                    'netG_state_dict': netG.module.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                    'schedularG_state_dict': schedulerG.state_dict(),
                    'netD_state_dict': netD.module.state_dict(),
                    'optimD_state_dict': optimD.state_dict(),
                    'schedularD_state_dict': schedulerD.state_dict(),
                }, os.path.join(model_ckpt_dir, f'{opt.name}_{str(total_step).zfill(len(str(opt.steps)))}.ckpt'))
                    
            if total_step==opt.steps:
                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.module.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                    'schedularG_state_dict': schedulerG.state_dict(),
                    'netD_state_dict': netD.module.state_dict(),
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
