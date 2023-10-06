import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

def smoothing(array, a):
    new_array = np.zeros_like(array)
    new_array[0] = array[0]
    for i in range(1, len(new_array)):
        new_array[i] = (1-a)*array[i] + a*new_array[i-1]
    return new_array

def plt_losses_train(log_names, csv_paths, columns_to_plot, out_path):
    plt.figure(figsize=(16, 9))
    for log_name, csv_path in zip(log_names, csv_paths):
        df = pd.read_csv(csv_path)
        X = df['total_step']
        for col_name in columns_to_plot:
            plt.plot(X, smoothing(df[col_name], a=0.9), label=f'{log_name}: {col_name}', alpha=0.5)

    plt.grid()
    plt.legend()
    plt.ylim(-5, 10)
    plt.xlabel('Step')
    plt.ylabel(f'{", ".join(columns_to_plot)}')
    plt.title(f'Value')
    plt.savefig(out_path)

def plt_losses_val(log_names, csv_paths, columns_to_plot, out_path):
    plt.figure(figsize=(8, 4.5))
    min_vals = []
    for log_name, csv_path in zip(log_names, csv_paths):
        df = pd.read_csv(csv_path)
        X = df['total_step']
        for col_name in columns_to_plot:
            plt.plot(X, smoothing(df[col_name], 0.0), label=f'{log_name}', alpha=0.5)
            min_idx = df[col_name].idxmin()
            min_step = df['total_step'][min_idx]
            min_val = df[col_name][min_idx]
            min_vals.append(str(min_val))
            plt.scatter(min_step, min_val, s=10, color='r')

    plt.grid()
    plt.legend(fontsize=6)
    #plt.ylim(50, 150)
    plt.xlabel('Step')
    plt.ylabel(f'{", ".join(columns_to_plot)}')
    plt.title(f'{", ".join(min_vals)}')
    plt.savefig(out_path)


if __name__=='__main__':
    whole_log_name, log_names = 'sep', ['TEST_ours_anime_sal2_lr', 'TEST_ours_anime_sal2_lr_sep', 'TEST_ours_anime_saladv_lr_recons', 'TEST_ours_anime_saladv_lr_sep_recons']

    #whole_log_name, log_names = 'clip', ['TEST_ours_anime_saladv_lr_sep_recons', 'TEST_ours_anime_saladv_lr_sep_recons_resume', 'TEST_ours_anime_saladv_lr_sep_recons_clip', 'TEST_ours_anime_saladv_lr_sep_recons_mod', 'TEST_ours_anime_saladv_lr_sep_recons_clip_mod', 'TEST_ours_anime_saladv_lr_sep_recons_clip_idt_mod']
    #whole_log_name, log_names = 'adv', ['TEST_ours_anime_sal2_lr', 'TEST_ours_anime_saladv_lr_recons', 'TEST_ours_anime_saladv_lr_sep_recons']
    whole_log_name, log_names = 'idt', ['TEST_ours_anime_saladv_lr_sep_recons_clip_mod', 'TEST_ours_anime_saladv_lr_sep_recons_clip_idt_mod', 'TEST_ours_anime_saladv_lr_sep_recons_idt_mod', 'TEST_ours_anime_saladv_lr_sep_recons_idt_mod_b4', 'TEST_ours_anime_saladv_lr_sep_recons_idt_mod_larger']
    #whole_log_name, log_names = 'mspc', ['MSPC_horse2zebra_b04_lr', 'MSPC_horse2zebra_b04_lr2', 'MSPC_horse2zebra_b04_lsgan']
    whole_log_name, log_names = 'h2z_ours', ['ours_horse2zebra_lr_sep_recons_idt', 'ours_horse2zebra_saladv_lr_sep_recons_idt', 'ours_horse2zebra_lr_sep_recons_idt10', 'ours_horse2zebra_lr_sep_recons10_idt', 'ours_horse2zebra_lr_sep_recons10_idt10', 'ours_horse2zebra_lr_sep_idt_foreonly', 'ours_horse2zebra_lr_sep_idt_foreonly2']
    whole_log_name, log_names = 'c2d_ours', ['ours_cat2dog_lr_sep_recons_idt', 'ours_cat2dog_saladv_lr_sep_recons_idt']
    whole_log_name, log_names = 'a2o_ours', ['ours_apple2orange_lr_sep_recons_idt', 'ours_apple2orange_saladv_lr_sep_recons_idt']
    whole_log_name, log_names = 's2a_ours', ['TEST_ours_anime_saladv_lr_sep_recons_idt_mod', 'ours_selfie2anime_lr_sep_recons_idt_resume', 'TEST_ours_anime_saladv_lr_sep_recons_idt_mod_b4']
    whole_log_name, log_names = 's2a_ours', ['TEST_ours_anime_saladv_lr_sep_recons_idt_mod', 'ours_selfie2anime_p3m10k', 'ours_selfie2anime_p3m10k_saladv', 'ours_selfie2anime_p3m10k_foreonly', 'ours_selfie2anime_p3m10k_foreonly_binary']
    whole_log_name, log_names = 'h2z_ours_foreonly', ['ours_horse2zebra_lr_sep_idt_foreonly2', 'ours_horse2zebra_lr_sep_idt_foreonly3', 'ours_horse2zebra_lr_sep_idt_foreonly4', 'ours_horse2zebra_lr_sep_idt_foreonly_multires']
    whole_log_name, log_names = 'a2o_ours_foreonly', ['TEST_ours_anime_saladv_lr_sep_recons_idt_mod', 'ours_selfie2anime_foreonly', 'ours_selfie2anime_foreonly_adv', 'ours_selfie2anime_foreonly_adv_mask', 'ours_selfie2anime_foreonly_adv_mask_recons', 'ours_selfie2anime_foreonly_adv_mask_recons_all', 'ours_selfie2anime_foreonly_all', 'ours_selfie2anime_foreonly_adv_mask_recons_all2', 'ours_selfie2anime_foreonly_adv_mask_recons_all3']
    whole_log_name, log_names = 'a2o_ours_foreonly', ['TEST_ours_anime_saladv_lr_sep_recons_idt_mod', 'ours_selfie2anime_foreonly', 'ours_selfie2anime_foreonly_adv_mask', 'ours_selfie2anime_foreonly_adv_mask_recons', 'ours_selfie2anime_foreonly_adv_mask_recons_all', 'ours_selfie2anime_foreonly_adv_mask_recons_all2', 'ours_selfie2anime_foreonly_adv_mask_recons_all3']
    whole_log_name, log_names = 'edges2shoes', ['MSPC_edges2shoes_b04_2', 'ours_edges2shoes_bi', 'ours_edges2shoes_oneside_idt10', 'ours_edges2shoes_oneside2', 'ours_edges2shoes_oneside3',  'ours_edges2shoes_oneside_b4']
    whole_log_name, log_names = 'edges2handbags', ['MSPC_edges2handbags_b04', 'ours_edges2handbags_bi', 'ours_edges2handbags_bi2', 'ours_edges2handbags_oneside_idt10', 'ours_edges2handbags_oneside2', 'ours_edges2handbags_oneside3', 'ours_edges2handbags_oneside_b4', 'ours_edges2handbags_oneside_b4_2']
    whole_log_name, log_names = 'edges2shoes', ['ours_edges2shoes_oneside2', 'ours_edges2shoes_oneside3', 'ours_edges2shoes_oneside4', 'ours_edges2shoes_oneside5', 'ours_edges2shoes_oneside6', 'ours_edges2shoes_oneside7']
    whole_log_name, log_names = 'a2o_ours_foreonly', ['ours_selfie2anime_idt_foreonly_adv_mask_recons_allonly', 'ours_selfie2anime_idt_foreonly_adv_mask_recons_allonly2', 'ours_selfie2anime_foreonly_adv_mask_recons_all2', 'ours_selfie2anime_idt_foreonly_adv_mask_recons_allonly2_resume', 'ours_selfie2anime_foreonly_adv_mask_recons_all4', 'ours_selfie2anime_foreonly_adv_mask_recons_all5', 'ours_selfie2anime_foreonly_adv_mask_recons_all4_resume', 'ours_selfie2anime_foreonly_adv_mask_recons_all4_resume2']
    whole_log_name, log_names = 'a2o_ours_foreonly', ['ours_selfie2anime_foreonly_adv_mask_recons_all4', 'ours_selfie2anime_foreonly_adv_mask_recons_all4_resume4',  'ours_selfie2anime_foreonly_adv_mask_recons_all4_resume4_resume2', 'ours_selfie2anime_foreonly_adv_mask_recons_all_b2_09', 'ours_selfie2anime_foreonly_adv_mask_recons_all_b2_09_2', 'ours_selfie2anime_foreonly_adv_mask_recons_all_multires']
    #whole_log_name, log_names = 'front', ['MSPC_front2side_b04_lsgan', 'MSPC_front2side_b04_lsgan2', 'ours_front2']

    test_only = True 

    if not test_only:
        columns_to_plot = ['loss_G', 'loss_D']
        train_csv_paths = []
        for log_name in log_names:
            log_path = f'experiments/{log_name}/logs'
            train_csv_paths.append(os.path.join(log_path, f'train_losses_{log_name}.csv'))
        out_path = os.path.join('comp', f'train_losses_{whole_log_name}.png')
        plt_losses_train(log_names, train_csv_paths, columns_to_plot, out_path)

    columns_to_plot = ['fid_score']
    val_csv_paths = []
    for log_name in log_names:
        log_path = f'experiments/{log_name}/logs'
        val_csv_paths.append(os.path.join(log_path, f'test_losses_{log_name}.csv'))
    out_path = os.path.join('comp', f'val_losses_{whole_log_name}.png')
    plt_losses_val(log_names, val_csv_paths, columns_to_plot, out_path)

