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

def plt_losses(csv_path, columns_to_plot, out_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 4.5))
    X = df['total_step']
    for col_name in columns_to_plot:
        plt.plot(X, smoothing(df[col_name], 0.9), label=f'{col_name}', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.savefig(out_path)

if __name__=='__main__':
    csv_path = 'temp2.csv'
    #columns_to_plot = ['loss_G_GA','loss_G_GB','loss_G_cycle_A','loss_G_cycle_B','loss_idt_A','loss_idt_B','loss_G','loss_D_B','loss_D_GA','loss_D_AB','loss_D_BA_A','loss_D_BA_GB','loss_D_BA']
    columns_to_plot = ['loss_G','loss_D_AB','loss_D_BA']
    out_path='temp.png'
    plt_losses(csv_path, columns_to_plot, out_path)
    csv_path = 'temp2.csv'
    #columns_to_plot = ['loss_G_GA','loss_G_GB','loss_G_cycle_A','loss_G_cycle_B','loss_idt_A','loss_idt_B','loss_G','loss_D_B','loss_D_GA','loss_D_AB','loss_D_BA_A','loss_D_BA_GB','loss_D_BA']
    columns_to_plot = ['loss_G','loss_D_AB','loss_D_BA']
    out_path='temp.png'
    plt_losses(csv_path, columns_to_plot, out_path)

