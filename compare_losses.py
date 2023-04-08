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
    plt.ylim(-2, 2)
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
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(f'{", ".join(columns_to_plot)}')
    plt.title(f'{", ".join(min_vals)}')
    plt.savefig(out_path)


if __name__=='__main__':
    #log_names = ['MSPC_anime', 'MSPC_anime_mod', 'MSPC_anime_mod_mod']
    #log_names = ['MSPC_horse2zebra_b01', 'MSPC_horse2zebra_b01_lsgan', 'MSPC_horse2zebra_b01_lsgan_idt']
    #log_names = ['MSPC_horse2zebra_b01', 'MSPC_horse2zebra_b01_lsgan', 'MSPC_horse2zebra_b01_lsgan_idt']
    log_names = ['MSPC_horse2zebra_b01_lsgan', 'MSPC_horse2zebra_b01_lsgan_idt', 'MSPC_horse2zebra_b01_lsgan_clipD', 'MSPC_horse2zebra_b01_lsgan_clipD_later']
    whole_log_name = 'MSPC_h2z_b01_lsgan'

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

