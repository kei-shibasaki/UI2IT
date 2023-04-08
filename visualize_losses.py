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

def plt_losses_train(csv_path, columns_to_plot, out_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 4.5))
    X = df['total_step']
    for col_name in columns_to_plot:
        plt.plot(X, smoothing(df[col_name], 0.9), label=f'{col_name}', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.savefig(out_path)

def plt_losses_val(csv_path, columns_to_plot, out_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 4.5))
    X = df['total_step']
    for col_name in columns_to_plot:
        plt.plot(X, df[col_name], label=f'{col_name}', alpha=0.5)
        min_idx = df[col_name].idxmin()
        min_step = df['total_step'][min_idx]
        min_val = df[col_name][min_idx]
        plt.scatter(min_step, min_val, s=10, color='r')

    plt.grid()
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(f'Min FID: {min_val:.3f}')
    plt.savefig(out_path)


if __name__=='__main__':
    log_name = 'TEST_MSPC_anime'
    log_path = f'experiments/{log_name}/logs'
    train_csv_path = os.path.join(log_path, f'train_losses_{log_name}.csv')
    val_csv_path = os.path.join(log_path, f'test_losses_{log_name}.csv')
    columns_to_plot = ['loss_G','loss_D']
    out_path = os.path.join(log_path, f'train_losses_{log_name}.png')
    plt_losses_train(train_csv_path, columns_to_plot, out_path)
    columns_to_plot = ['fid_score']
    out_path = os.path.join(log_path, f'val_losses_{log_name}.png')
    plt_losses_val(val_csv_path, columns_to_plot, out_path)

