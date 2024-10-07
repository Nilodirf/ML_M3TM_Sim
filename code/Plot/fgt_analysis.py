import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import io

from ..Source.finderb import finderb


def get_data(file, subsys):

    sim_files_folder = 'Results/FGT/' + file + '/'
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

    axs[0].set_ylabel(r'$T_e$', fontsize=16)
    axs[1].set_ylabel(r'$m/m_0$', fontsize=16)
    axs[2].set_ylabel(r'MSD', fontsize=16)

    axs[0].set_xlim(-0.05, 0.5)
    axs[1].set_xlim(-0.1, 10)
    axs[2].set_xlim(-0.1, 10)

    axs[1].set_ylim(0.5, 1.05)

    fig.tight_layout()

    return fig, axs


def plot_te(file, figure, axs, show_exp=True):
    delay, te, bla = get_data(file, 'te')
    axs[0].plot(delay, te, color='orange')

    if show_exp:
        exp_delay, f1, df1 = get_te_exp()
        axs[0].errorbar(exp_delay, f1, yerr=df1, fmt='o', color='orange')

    return figure, axs


def plot_mag(file, figure, axs, show_exp=True):
    delay, mag, bla = get_data(file, 'mag')
    mag = 2/3 + mag/3
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
    ep = cp*tp[:, 0]
    ep_norm = (ep - ep[0]) / ep[finderb(5, delay)[0]] * 0.4
    axs[2].plot(delay, ep_norm, color='blue', alpha=0.3)

    if tp2 is not None:
        cp2_dat = np.loadtxt('input_data/FGT/FGT_c_p2.txt')
        temp2 = cp2_dat[:, 0]
        cp2_temp = cp2_dat[:, 1]
        temp2_indices = finderb(tp2, temp2)
        cp2 = cp2_temp[temp2_indices]
        ep2 = cp2 * tp2[:, 0]
        ep2_norm = (ep2-ep2[0]) / ep2[finderb(5, delay)[0]] * 0.6

        ep_tot_norm = ep_norm + ep2_norm/(np.amax(ep_norm+ep2_norm))

        axs[2].plot(delay, ep2_norm, color='blue', ls='dashed', alpha=0.2)
        axs[2].plot(delay, ep_tot_norm, color='blue')

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


def init_fit():
    # Find the optimal parameters for \gamma and therm_time from fitting the initial electron temperature increase

    files = os.listdir('Results/FGT/fits_init')
    gammas = np.arange(180, 241)
    therm_times = np.arange(0, 41)
    chi_sq = np.zeros((41, 61))
    for file in files:
        full_path = 'fits_init/' + file
        therm_time = float(file[:file.find('_')])
        gamma = int(file[file.find('_')+1:])

        therm_time_index = finderb(therm_time, therm_times)[0]
        gamma_index = finderb(gamma, gammas)[0]

        delay, te, bla = get_data(full_path, 'te')
        exp_delay, exp_te, exp_dte = get_te_exp()

        max_fit_delay_index = finderb(0.045, exp_delay)[0]+3
        fit_delay = exp_delay[:max_fit_delay_index]

        delay_indices = finderb(fit_delay, delay)
        cs = np.sum(((exp_te[:max_fit_delay_index]-te[delay_indices])/exp_dte[:max_fit_delay_index])**2)
        cs_norm = cs/len(delay_indices)*np.sum(exp_dte[:max_fit_delay_index]**(-2))

        chi_sq[therm_time_index, gamma_index] = cs_norm

    min_ind = np.argmin(chi_sq)
    tt_fit_ind = int((min_ind - (int(min_ind) % len(therm_times))/len(therm_times)))
    gamma_fit_ind = int(int(min_ind) % len(gammas))

    print('min = ', str(chi_sq[tt_fit_ind, gamma_fit_ind]))
    print('tt_fit = ', str(therm_times[tt_fit_ind]) + ' fs')
    print('gamma_fit = ', str(gammas[gamma_fit_ind]) + ' J/m^3/K^2')

    therm_times, gammas = np.meshgrid(therm_times, gammas)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(therm_times, gammas, chi_sq.T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    return


def show_fits(save, show):
    fig, axs = create_figure()
    fig, axs = plot_te(file='fits_new/te_tt_15fs', figure=fig, axs=axs)
    fig, axs = plot_tp(file='fits_new/tp_tt_15fs', figure=fig, axs=axs)
    fig, axs = plot_mag(file='fits_new/mag_tt_15fs', figure=fig, axs=axs)
    if save:
        save_plot(fig, axs, 'bad_fit_to_will.pdf')
    if show:
        show_plot(fig, axs)

        return


init_fit()
