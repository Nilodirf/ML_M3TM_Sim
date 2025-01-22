import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import io


def finderb(key, array):

    key = np.array(key, ndmin=1)
    n = len(key)
    i = np.zeros([n], dtype=int)

    for m in range(n):
        i[m] = finderb_nest(key[m], array)
    return i


def finderb_nest(key, array):

    a = 0  # start of intervall
    b = len(array)  # end of intervall

    # if the key is smaller than the first element of the
    # vector we return 1
    if key < array[0]:
        return 0

    while (b-a) > 1:  # loop until the intervall is larger than 1
        c = int(np.floor((a+b)/2))  # center of intervall
        if key < array[c]:
            # the key is in the left half-intervall
            b = c
        else:
            # the key is in the right half-intervall
            a = c

    return a


def get_data(file, subsys):

    sim_files_folder = file + '/'
    delay = np.load(sim_files_folder + 'delay.npy')
    if subsys == 'te':
        val_1 = np.load(sim_files_folder + 'tes.npy')[:, 0]
        val_2 = None
    elif subsys == 'tp':
        val_1 = np.load(sim_files_folder + 'tps.npy')[:, 0]
        if os.path.isfile(sim_files_folder + 'tp2s.npy'):
            val_2 = np.load(sim_files_folder + 'tp2s.npy')[:, 0]
        else:
            val_2 = None
    elif subsys == 'mag':
        val_1 = np.load(sim_files_folder + 'ms.npy')[:, 0]
        val_1 = val_1/val_1[0]
        val_2 = None
    else:
        print('Idiot, put te ot tp or mag for subsys')
        return

    return delay*1e12-1, val_1, val_2


def get_te_exp():
    f = io.loadmat('input_data/FGT/exp_data/Temperatures.mat')

    exp_delay = f['Delay'] / 1e3

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

    return exp_delay, f1, df1


def get_mag_exp():
    exp_dat = np.loadtxt('input_data/FGT/exp_data/mag.txt')
    exp_delay = exp_dat[:, 0]
    exp_mag = np.copy(exp_dat[:, 1])
    time_zero_index = finderb(0, exp_delay)[0]
    exp_mag = exp_mag / exp_mag[time_zero_index]
    exp_dmag = exp_dat[:, 2] / exp_dat[time_zero_index, 1]

    return exp_delay, exp_mag, exp_dmag


def get_msd_exp():
    exp_dat = np.loadtxt('input_data/FGT/exp_data/MSD_SEP_24.txt')
    exp_delay = exp_dat[:, 0] - 0.7
    exp_msd = np.copy(exp_dat[:, 1])
    exp_msd /= np.amax(exp_msd)
    exp_dmsd = exp_dat[:, 2] / np.amax(exp_dat[:, 1])

    return exp_delay, exp_msd, exp_dmsd


def create_figure():

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))

    axs[2].set_xlabel(r'delay [ps]', fontsize=16)

    axs[0].set_ylabel(r'$T_e$ [K]', fontsize=16)
    axs[1].set_ylabel(r'$m_{tot}/m_0$', fontsize=16)
    axs[2].set_ylabel(r'$\Delta \widetilde{MSD}$', fontsize=16)

    axs[0].set_xlim(-0.05, 0.5)
    axs[1].set_xlim(-0.1, 5)
    axs[2].set_xlim(-0.1, 5)

    axs[1].set_ylim(0.5, 1.05)

    fig.tight_layout()

    return fig, axs

def create_figure_te(show, save):

    # create figure:
    plt.figure(figsize=(7, 6))
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,3)
    ax3 = plt.subplot(1,2,2)

    # boundaries:
    ax1.set_xlim(-0.5, 2)
    ax2.set_xlim(-0.5, 2)
    ax3.set_xlim(-0.5, 2)

    ax1.set_ylim(80, 920)
    ax2.set_ylim(80, 920)
    ax3.set_ylim(80, 920)

    # a, b, c labels:
    ax1.text(1, 600, r"a)", fontsize=20)
    ax2.text(1, 600, r"b)", fontsize=20)
    ax3.text(1, 600, r"c)", fontsize=20)

    # axes label adjustment:
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.set_xlabel(r"delay [ps]" , fontsize=18)
    ax3.set_ylabel(r"$T_e$ [K]", fontsize=18)
    ax1.set_xticklabels([])

    # experimental te data:
    exp_delay, f1, df1 = get_te_exp()

    # sim data of optimal fit:
    delay_fit, te_fit, _ = get_data(file='input_data/FGT/fit_results/a0.017gep5.1gpp3.9gamma215.0_el', subsys='te')
    delay_fit_tp, tp_fit, tp2_fit = get_data(file='input_data/FGT/fit_results/a0.017gep5.1gpp3.9gamma215.0_el', subsys='tp')

    # sim data of high gamma:
    delay_gamma, te_gamma, _ = get_data(file='Results/FGT/fits_new/te_gamma_high', subsys='te')
    delay_gamma_tp, tp_gamma, tp2_gamma = get_data(file='Results/FGT/fits_new/te_gamma_high', subsys='tp')

    # sim data of high fluence:
    delay_flu, te_flu, _ = get_data(file='Results/FGT/fits_new/te_flu_high', subsys='te')
    delay_flu, tp_flu, tp2_flu = get_data(file='Results/FGT/fits_new/te_flu_high', subsys='tp')

    # high gamma plot:
    ax1.plot(delay_gamma, te_gamma, lw=2.0, color='orange')
    ax1.plot(delay_gamma, tp_gamma, lw=2.0, ls ='dashed', color='blue')
    ax1.plot(delay_gamma, tp2_gamma, lw=2.0, ls ='dashed', color='lightblue')
    ax1.errorbar(exp_delay, f1, yerr=df1, fmt='o', color='orange')

    # high fluence plot:
    ax2.plot(delay_flu, te_flu, lw=2.0, color='orange')
    ax2.plot(delay_flu, tp_flu, lw=2.0, ls ='dashed', color='blue')
    ax2.plot(delay_flu, tp2_flu, lw=2.0, ls ='dashed', color='lightblue')
    ax2.errorbar(exp_delay, f1, yerr=df1, fmt='o', color='orange')

    # high gamma plot:
    ax3.plot(delay_fit, te_fit, lw=2.0, color='orange')
    ax3.plot(delay_fit_tp, tp_fit, lw=2.0, ls ='dashed', color='blue')
    ax3.plot(delay_fit_tp, tp2_fit, lw=2.0, ls ='dashed', color='lightblue')
    ax3.errorbar(exp_delay, f1, yerr=df1, fmt='o', color='orange')

    # labels in ax3:
    ax3.legend([r'$T_e$', r'$T_{p,o}$', r'$T_{p, a}$'], fontsize=16, loc='upper right')

    if save:
        plt.savefig('new_te_figure.pdf')
    if show:
        plt.show()

    return


def plot_te(file, figure, axs, show_exp=True):
    delay, te, bla = get_data(file, 'te')
    delay += 0.001
    axs[0].plot(delay, te, color='orange')

    if show_exp:
        exp_delay, f1, df1 = get_te_exp()
        axs[0].errorbar(exp_delay, f1, yerr=df1, fmt='o', color='orange')

    return figure, axs


def plot_mag(file, figure, axs, show_exp=True):
    delay, mag, bla = get_data(file, 'mag')
    mag = 0.73 + mag*0.27
    axs[1].plot(delay, mag, color='green')

    if show_exp:
        exp_delay, exp_mag, exp_dmag = get_mag_exp()
        axs[1].errorbar(exp_delay, exp_mag, yerr=exp_dmag, fmt='o', color='green')

    return figure, axs


def plot_tp(file, figure, axs, show_exp=True):
    delay, tp, tp2 = get_data(file, 'tp')

    cp_dat = np.loadtxt('input_data/FGT/FGT_c_p1.txt')
    temp = cp_dat[:, 0]
    cp_temp = cp_dat[:, 1]
    temp_indices = finderb(tp, temp)
    cp = cp_temp[temp_indices]
    ep = cp*tp
    ep_norm = (ep - ep[0]) / ep[finderb(5, delay)[0]]

    if tp2 is not None:
        cp2_dat = np.loadtxt('input_data/FGT/FGT_c_p2.txt')
        temp2 = cp2_dat[:, 0]
        cp2_temp = cp2_dat[:, 1]
        temp2_indices = finderb(tp2, temp2)
        cp2 = cp2_temp[temp2_indices]
        ep2 = cp2 * tp2
        ep2_norm = (ep2-ep2[0]) / ep2[finderb(5, delay)[0]] * 0.7

        ep_norm = ep_norm * 0.3
        ep_tot_norm = ep_norm + ep2_norm/(np.amax(ep_norm+ep2_norm))

        axs[2].plot(delay, ep2_norm, color='blue', ls='dashed', alpha=0.2)
        axs[2].plot(delay, ep_tot_norm, color='blue')

    axs[2].plot(delay, ep_norm, color='blue', alpha=0.3)

    if show_exp:
        exp_delay, exp_msd, exp_dmsd = get_msd_exp()
        axs[2].errorbar(exp_delay, exp_msd, yerr=exp_dmsd, fmt='o', color='blue')

    return figure, axs


def show_plot(figure, axs):
    plt.show()

    return


def save_plot(figure, axs, name):
    plt.savefig(name)

    return


def show_fits(show, save):
    fig, axs = create_figure()
    # fig, axs = plot_te(file='input_data/FGT/fit_results/a0.017gep5.1gpp3.9gamma215.0_el', figure=fig, axs=axs)
    # fig, axs = plot_tp(file='input_data/FGT/fit_results/a0.017gep5.1gpp3.9gamma215.0_tp', figure=fig, axs=axs)
    # fig, axs = plot_mag(file='input_data/FGT/fit_results/a0.017gep5.1gpp3.9gamma215.0_mag', figure=fig, axs=axs)

    fig, axs = plot_te(file='input_data/FGT/fit_results/a0.02gep5.1gpp3.9gamma213.0_el', figure=fig, axs=axs)
    fig, axs = plot_tp(file='input_data/FGT/fit_results/a0.02gep5.1gpp3.9gamma213.0_tp', figure=fig, axs=axs)
    fig, axs = plot_mag(file='input_data/FGT/fit_results/a0.02gep5.1gpp3.9gamma213.0_mag', figure=fig, axs=axs)

    if save:
        save_plot(fig, axs, 'new_fit_to_show.pdf')
    if show:
        show_plot(fig, axs)

        return


if __name__ == "__main__":
    # create_figure_te(True, False)
    show_fits(True, False)
