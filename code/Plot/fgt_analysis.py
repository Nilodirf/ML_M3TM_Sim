import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import io
from scipy.stats import chi2

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
    ep = cp*tp
    ep_norm = (ep - ep[0]) / ep[finderb(5, delay)[0]] * 0.4
    axs[2].plot(delay, ep_norm, color='blue', alpha=0.3)

    if tp2 is not None:
        cp2_dat = np.loadtxt('input_data/FGT/FGT_c_p2.txt')
        temp2 = cp2_dat[:, 0]
        cp2_temp = cp2_dat[:, 1]
        temp2_indices = finderb(tp2, temp2)
        cp2 = cp2_temp[temp2_indices]
        ep2 = cp2 * tp2
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


def show_fits(save, show):
    fig, axs = create_figure()
    fig, axs = plot_te(file='fits_new/te_tt_15fs', figure=fig, axs=axs)
    fig, axs = plot_tp(file='fits_new/tp_tt_15fs', figure=fig, axs=axs)
    fig, axs = plot_mag(file='fits_new/mag_tt_15fs', figure=fig, axs=axs)
    if save:
        save_plot(fig, axs, 'bad_fit_to_show.pdf')
    if show:
        show_plot(fig, axs)

        return


show_fits(save=True, show=True)


def init_fit():
    # Find the optimal parameters for \gamma and therm_time from fitting the initial electron temperature increase

    files = os.listdir('Results/FGT/fits_init')
    gammas = np.arange(165, 256)
    therm_times = np.arange(0, 41)
    chi_sq = np.ones((41, 91))*100
    for file in files:
        full_path = 'fits_init/' + file
        therm_time = float(file[:file.find('_')])
        gamma = int(file[file.find('_')+1:])

        therm_time_index = finderb(therm_time, therm_times)[0]
        gamma_index = finderb(gamma, gammas)[0]

        delay, te, bla = get_data(full_path, 'te')
        exp_delay, exp_te, exp_dte = get_te_exp()

        max_fit_delay_index = finderb(20, exp_delay)[0]
        fit_delay = exp_delay[:max_fit_delay_index]

        delay_indices = finderb(fit_delay, delay)
        cs = np.sum(((exp_te[:max_fit_delay_index]-te[delay_indices])/exp_dte[:max_fit_delay_index])**2)
        cs_norm = cs/len(delay_indices)

        # if therm_time == 13 and gamma == 211:
        #     plt.figure(figsize=(8, 6))
        #     plt.plot(delay[:delay_indices[-1]], te[:delay_indices[-1]], color='orange', label='sim')
        #     plt.scatter(delay[delay_indices], te[delay_indices], color='orange', label='sim points')
        #     plt.errorbar(exp_delay[:max_fit_delay_index], exp_te[:max_fit_delay_index],
        #                  yerr=exp_dte[:max_fit_delay_index],
        #                  fmt='o', color='blue', label='data points')
        #     plt.legend(fontsize=14)
        #     plt.xlabel(r'delay [ps]', fontsize=16)
        #     plt.ylabel(r'$T_e$ [K]', fontsize=16)
        #     plt.xlim(delay[delay_indices[0]], delay[delay_indices[-1]])
        #     plt.savefig('Results/FGT/best_init_fit.pdf')
        #     plt.show()

        chi_sq[therm_time_index, gamma_index] = cs_norm

    min = np.amin(chi_sq)
    conf = chi2.ppf(min, df=2)
    p = chi2.ppf(0.68, df=2)
    min_ind = np.argmin(chi_sq)

    tt_fit_ind, gamma_fit_ind = np.unravel_index(min_ind, chi_sq.shape)
    tt_fit = therm_times[tt_fit_ind]
    gamma_fit = gammas[gamma_fit_ind]

    gamma_konv_ind = finderb(min+p, chi_sq[tt_fit_ind, :])
    tt_konv_ind = finderb(min+p, chi_sq[:, gamma_fit_ind])
    gamma_konv = gammas[gamma_konv_ind]
    tt_konv = therm_times[tt_konv_ind]
    sigma_tt = np.abs(tt_fit-tt_konv)/p
    sigma_gamma = np.abs(gamma_fit-gamma_konv)/p

    print('++++++++++++++++++++++++++++++++')
    print('init_fit')
    print('confidence: ', conf)
    print('minimum of chi_sq: ', min)
    print('best tt fit value: ', tt_fit, ' fs')
    print('best gamma fit value: ', gamma_fit, ' J/m^3/K^2')
    print('sigma tt: ', sigma_tt, ' fs')
    print('sigma gamma: ', sigma_gamma, ' J/m^3/K^2')
    print('++++++++++++++++++++++++++++++++')
    print()

    therm_times, gammas = np.meshgrid(therm_times, gammas)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(therm_times, gammas, chi_sq.T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5, label=r'$\chi^2$')
    plt.xlabel(r'thermalization time [fs]', fontsize=14)
    plt.ylabel(r'$\gamma_{el}$ [J/m$^3$/K$^2$]', fontsize=14)
    plt.show()

    return


def inter_fit(show_fit=False, show_asf=0, show_gep=0):
    files_te = os.listdir('Results/FGT/fits_inter_2/el')
    files_mag = os.listdir('Results/FGT/fits_inter_2/mag')
    folder_str = ['el/', 'mag/']
    asfs = np.array([0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025])
    geps = np.array([4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4,
                     5.5, 5.6, 5.7, 5.8, 5.9, 6., 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.])
    chi_sq_te = np.zeros((16, 31))
    chi_sq_mag = np.zeros((16, 31))

    for folder, f_str, cs in zip([files_te, files_mag], folder_str, [chi_sq_te, chi_sq_mag]):
        for file in folder:
            full_path = 'fits_inter_2/' + f_str + file
            asf = float(file[1:file.find('g')])
            gep = float(file[file.find('g') + 1:])

            asf_index = finderb(asf, asfs)[0]
            gep_index = finderb(gep, geps)[0]

            if f_str == 'el/':
                delay, dat, bla = get_data(full_path, 'te')
                exp_delay, exp_dat, exp_dd = get_te_exp()
                dd_sq_el = np.sum((exp_dd/exp_dat)**2)
                n_el = len(exp_dat)
                n = n_el
            else:
                delay, dat, bla = get_data(full_path, 'mag')
                dat = 2/3 + 1/3*dat
                exp_delay, exp_dat, exp_dd = get_mag_exp()
                dd_sq_mag = np.sum((exp_dd / exp_dat) ** 2)
                n_mag = len(exp_dat)
                n = n_mag

            delay_indices = finderb(exp_delay, delay)
            cs_norm = np.sum(((exp_dat - dat[delay_indices]) / exp_dd) ** 2)/ len(delay_indices-2)
            cs[asf_index, gep_index] = cs_norm

            if show_fit:
                if gep == show_gep and asf == show_asf:
                    plt.figure(figsize=(8, 6))
                    plt.plot(delay[:delay_indices[-1]], dat[:delay_indices[-1]], color='orange', label='sim')
                    plt.scatter(delay[delay_indices], dat[delay_indices], color='orange', label='sim points')
                    plt.errorbar(exp_delay, exp_dat,
                                 yerr=exp_dd,
                                 fmt='o', color='blue', label='data points')
                    plt.legend(fontsize=14)
                    plt.xlabel(r'delay [ps]', fontsize=16)
                    plt.ylabel(r'Observable', fontsize=16)
                    plt.xlim(delay[delay_indices[0]], delay[delay_indices[-1]])
                    # plt.savefig('Results/FGT/best_init_fit.pdf')
                    plt.show()

        min = np.amin(cs)
        conf = chi2.cdf(min, df=n-2)
        p = chi2.ppf(0.68, df=n-2)
        min_ind = np.argmin(cs)

        asf_fit_ind, gep_fit_ind = np.unravel_index(min_ind, cs.shape)
        asf_fit = asfs[asf_fit_ind]
        gep_fit = geps[gep_fit_ind]

        asf_konv_ind = finderb(min + p, cs[:, gep_fit_ind])
        gep_konv_ind = finderb(min + p, cs[asf_fit_ind, :])
        asf_konv = asfs[asf_konv_ind]
        gep_konv = geps[gep_konv_ind]
        sigma_asf = np.abs(asf_fit - asf_konv) / p
        sigma_gep = np.abs(gep_fit - gep_konv) / p

        print('++++++++++++++++++++++++++++++++')
        print('inter_fit')
        print('subssystem: ', f_str)
        print('confidence: ', conf)
        print('within 0.68 probability: ', p)
        print('minimum of chi_sq: ', min)
        print('best asf fit value: ', asf_fit)
        print('best gep fit value: ', gep_fit, ' W/m^3/K')
        print('sigma asf: ', sigma_asf)
        print('sigma gep: ', sigma_gep, ' W/m^3/K')
        print('++++++++++++++++++++++++++++++++')
        print()

        asfs_mesh, geps_mesh = np.meshgrid(asfs, geps)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(asfs_mesh, geps_mesh, cs.T, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5, label=r'$\chi^2$')
        plt.xlabel(r'$a_{sf}$', fontsize=14)
        plt.ylabel(r'$g_{ep}$ [W/m$^3$/K]', fontsize=14)
        plt.title(f_str, fontsize=16)
        plt.show()

    # determine and normalize the overall fit quality of two subsystems:
    chi_sq_te_weighted = chi_sq_te * (n_el-2)
    chi_sq_mag_weighted = chi_sq_mag * (n_mag-2)
    chi_sq_tot = (chi_sq_te_weighted + chi_sq_mag_weighted) / (n_el+n_mag-4)

    # find the minimum and confidence of the fit, and the confidence radius of sigma:
    min = np.amin(chi_sq_tot)
    conf = chi2.cdf(min, df=n_el+n_mag-4)
    p = chi2.ppf(0.68, df=n_el+n_mag-4)
    min_ind = np.argmin(chi_sq_tot)

    # find the corresponding fit parameter values for the best fit:
    asf_fit_ind, gep_fit_ind = np.unravel_index(min_ind, chi_sq_tot.shape)
    asf_fit = asfs[asf_fit_ind]
    gep_fit = geps[gep_fit_ind]

    # find the short and long axis of the ellipsis (not guaranteed that this is right!):
    asf_konv_ind = finderb(min + p, chi_sq_tot[:, gep_fit_ind])
    gep_konv_ind = finderb(min + p, chi_sq_tot[asf_fit_ind, :])
    asf_konv = asfs[asf_konv_ind]
    gep_konv = geps[gep_konv_ind]
    sigma_asf = np.abs(asf_fit - asf_konv) / p
    sigma_gep = np.abs(gep_fit - gep_konv) / p

    # find the proper ellipsis:
    conv_int_mask = np.abs(chi_sq_tot-np.amin(chi_sq_tot)-p) <= 5e-1
    conv_intervall = chi_sq_tot[conv_int_mask]
    conv_int_int = conv_int_mask.astype(int)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(asfs_mesh, geps_mesh, conv_int_int.T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5, label=r'$\chi^2$')
    plt.xlabel(r'$a_{sf}$', fontsize=14)
    plt.ylabel(r'$g_{ep}$ [W/m$^3$/K]', fontsize=14)
    plt.title('conv_rad', fontsize=16)
    plt.show()


    print('++++++++++++++++++++++++++++++++')
    print('inter_fit')
    print('subssystem: both')
    print('commulative porbability of fit quality: ', conf)
    print('chi2 within 0.68 probability: ', p)
    print('minimum of chi_sq: ', min)
    print('best asf fit value: ', asf_fit)
    print('best gep fit value: ', gep_fit, ' W/m^3/K')
    print('sigma asf: ', sigma_asf)
    print('sigma gep: ', sigma_gep, ' W/m^3/K')
    print('++++++++++++++++++++++++++++++++')
    print()

    asfs_mesh, geps_mesh = np.meshgrid(asfs, geps)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(asfs_mesh, geps_mesh, chi_sq_tot.T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5, label=r'$\chi^2$')
    plt.xlabel(r'$a_{sf}$', fontsize=14)
    plt.ylabel(r'$g_{ep}$ [W/m$^3$/K]', fontsize=14)
    plt.title(r'both', fontsize=16)
    plt.show()

    return

def global_manual_fit():
    files_te = os.listdir('Results/FGT/fits_global/el')
    files_mag = os.listdir('Results/FGT/fits_global/mag')
    files_tp = os.listdir('Results/FGT/fits_global/tp')
    folder_str = ['el/', 'mag/', 'tp/']

    gammas = np.arange(205, 226).astype(float)  # 20
    t0_el = np.arange(20)  # 20
    geps = np.array([4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5])  # 16
    asfs = np.array([0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02])  # 11
    gpps = np.array([2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0])  # 16

    chi_sq_te = np.zeros((20, 20, 16, 11, 16))  # gamma, t0, gep, asf, gpp
    chi_sq_mag = np.zeros((20, 20, 16, 11, 16))  # gamma, t0, gep, asf, gpp
    chi_sq_tp = np.zeros((20, 20, 16, 11, 16))  # gamma, t0, gep, asf, gpp

    for folder, f_str, cs in zip([files_te, files_mag], folder_str, [chi_sq_te, chi_sq_mag]):
        for file in folder:
            full_path = 'fits_inter_2/' + f_str + file
            asf = float(file[1:file.find('g')])
            gep = float(file[file.find('g') + 1:])

            asf_index = finderb(asf, asfs)[0]
            gep_index = finderb(gep, geps)[0]



