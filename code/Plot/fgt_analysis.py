import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import io

from ..Source.finderb import finderb


def get_data(file, subsys):

    sim_files_folder = 'Results/FGT/' + file + '/'
    delay = np.load(sim_files_folder + 'delay.npy')
    if subsys == 'te':
        val_1 = np.load(sim_files_folder + 'tes.npy')
        val_2 = None
    elif subsys == 'tp':
        val_1 = np.load(sim_files_folder + 'tps.npy')
        if os.path.isfile(sim_files_folder + 'tp2s.npy'):
            val_2 = np.load(sim_files_folder + 'tp2s.npy')
        else:
            val_2 = None
    elif subsys == 'mag':
        val_1 = np.load(sim_files_folder + 'ms.npy')
        val_1 = val_1/val_1[0]
        val_2 = None
    else:
        print('Idiot, put te ot tp or mag for subsys')
        return

    return delay*1e12-1, val_1, val_2


def create_figure():

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,6))

    axs[2].set_xlabel(r'delay [ps]', fontsize=16)

    axs[0].set_ylabel(r'$m/m_0$', fontsize=16)
    axs[1].set_ylabel(r'$T_p$', fontsize=16)
    axs[2].set_ylabel(r'$T_e$', fontsize=16)

    axs[0].set_xlim(-0.1, 5)
    axs[1].set_xlim(-0.1, 5)
    axs[2].set_xlim(-0.1, 5)

    fig.tight_layout()

    return fig, axs


def plot_te(file, figure, axs, show_exp=True):
    delay, te, bla = get_data(file, 'te')
    axs[2].plot(delay, te, color='orange')

    if show_exp:
        f = io.loadmat('input_data/FGT/exp_data/Temperatures.mat')

        exp_delay = f['Delay']/1e3

        f1 = f['F0p6'][:, 0]
        # f2 = f['F1p2']
        # f3 = f['F1p8']
        # f4 = f['F2p5']
        # f5 = f['F3p0']
        df1 = f['Err_F0p6'][:, 0]
        # df2 = f['Err_F1p2']
        # df3 = f['Err_F1p8']
        # df4 = f['Err_2p5']
        # df5 = f['Err_F3p0']
        axs[2].errorbar(exp_delay, f1, yerr=df1, fmt='o', color='orange')

    return figure, axs


def plot_mag(file, figure, axs, show_exp=True):
    delay, mag, bla = get_data(file, 'mag')
    axs[0].plot(delay, mag, color='green')

    if show_exp:
        exp_dat = np.loadtxt('input_data/FGT/exp_data/mag.txt')
        exp_delay = exp_dat[:, 0]-0.1992
        exp_mag = exp_dat[:, 1]
        exp_mag = exp_mag/exp_mag[0]+0.1
        exp_dmag = exp_dat[:, 2]/0.017715
        axs[0].errorbar(exp_delay, exp_mag, yerr=exp_dmag, fmt='o', color='green')

    return figure, axs


def plot_tp(file, figure, axs, show_exp=True):
    delay, tp, tp2 = get_data(file, 'tp')

    cp_dat = np.loadtxt('input_data/FGT/FGT_c_p1.txt')
    temp = cp_dat[:, 0]
    cp_temp = cp_dat[:, 1]
    temp_indices = finderb(tp, temp)
    cp = cp_temp[temp_indices]
    ep = cp*tp[:, 0]
    axs[1].plot(delay, ep, color='blue')

    if tp2 is not None:
        cp2_dat = np.loadtxt('input_data/FGT/FGT_c_p2.txt')
        temp2 = cp2_dat[:, 0]
        cp2_temp = cp2_dat[:, 1]
        temp2_indices = finderb(tp2, temp2)
        cp2 = cp2_temp[temp2_indices]
        ep2 = cp2 * tp2[:, 0]
        axs[1].plot(delay, ep2, color='blue', ls='dashed')

    if show_exp:
        exp_dat = np.loadtxt('input_data/FGT/exp_data/MSD_SEP_24.txt')
        exp_delay = exp_dat[:, 0]
        exp_msd = exp_dat[:, 1]
        exp_dmsd = exp_dat[:, 2]
        axs[1].errorbar(exp_delay, exp_msd, yerr=exp_dmsd, fmt='o', color='blue')

    return figure, axs


def show_plot(figure, axs):
    plt.show()

    return


fig, axs = create_figure()
fig, axs = plot_te(file='fits_old/te', figure=fig, axs=axs)
fig, axs = plot_tp(file='fits_old/tp', figure=fig, axs=axs)
fig, axs = plot_mag(file='fits_old/mag', figure=fig, axs=axs)
show_plot(fig, axs)
