import numpy as np
import scipy
import math
import warnings

from matplotlib import pyplot as plt
from scipy import constants as sp
from scipy import interpolate as ip

from .plot import SimComparePlot
from .plot import SimPlot
from ..Source.finderb import finderb
from ..Source.finderb import finder_nosort


class SimAnalysis(SimComparePlot):
    def __init__(self, files):
        super().__init__(files)

    @staticmethod
    def mag_tem_plot(file):
        delays, tes, tps, mags = SimPlot(file).get_data()[:4]

        dz = 2e-9
        pen_dep = 14e-9
        exp_decay = np.exp(-np.arange(len(mags[0])) * dz / pen_dep)



        mag_av = np.sum(mags * exp_decay, axis=1) / np.sum(exp_decay)
        # mag_av /= np.amax(np.abs(mag_av-mag_av[0]))
        # te_av = np.sum(tes, axis=1)/len(mags[0])

        mmin_index = finder_nosort(np.amin(mag_av), mag_av)[0]

        tp_cgt_av = np.sum(tps[:, 8:16] * exp_decay, axis=1)/np.sum(exp_decay)
        tp_hbn_av = np.sum(tps[:, :8], axis=1)/8
        tp_sio2_av = np.sum(tps[:, 16:], axis=1)/len(tps.T[16:])

        fig = plt.figure(layout='constrained', figsize=(7, 9))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        ax1.plot(delays*1e9, tp_hbn_av, color=(237/255, 197/255, 171/255), label=r'hBN', lw=3.0)
        ax1.plot(delays*1e9, tp_cgt_av, color=(188/255, 123/255, 119/255), label=r'CGT', lw=3.0)
        ax1.plot(delays*1e9, tp_sio2_av, color=(33/255, 112/255, 116/255), label=r'SiO2', lw=3.0)

        ax1.hlines(65, -0.1, 2.01, lw=2.0, color='black', ls='dashed')
        ax1.text(-0.1, 67, r'$T_C$', fontsize=18)
        ax1.vlines(delays[mmin_index]*1e9, 6, 100, color='black', alpha=0.5, ls='dotted', lw=2.0)

        ax1.set_ylabel(r'Phonon temperature [K]', fontsize=20)
        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.tick_params(axis='y', labelsize=18)
        ax1.legend(fontsize=18)
        ax1.set_xlim(-0.1, 2.01)
        ax1.set_ylim(6, 100)


        ax2.plot(delays*1e9, mag_av, color=(83/255, 42/255, 42/255), lw=3.0)

        ax2.vlines(delays[mmin_index]*1e9, 0, 1, color='black', alpha=0.5, ls='dotted', lw=2.0)

        ax2.set_ylabel(r'weighted magnetization', fontsize=20)
        ax2.set_xlabel(r'delay [ns]', fontsize=20)
        ax2.set_xlim(-0.1, 2.01)
        ax2.set_ylim(0.2, 1.01)
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)

        # plt.plot(te_rec, mag(te_rec, 65), label=r'$m(T)=\sqrt{1-\frac{T}{T_c}}$')
        # plt.xlabel('temperature', fontsize=16)
        # plt.ylabel('magnetization', fontsize=16)
        # plt.title('Magnetization over Temperature', fontsize=18)
        # plt.legend(fontsize=14)
        #
        # plt.xscale('linear')
        # plt.gca().invert_xaxis()
        plt.savefig('mag_tem_cgt_thin.pdf')
        plt.show()
        return

    def plot_dmdt(self):
        m = np.linspace(0, 1, num=100)
        tem = np.arange(0, 300)

        R_CGT = SimAnalysis.get_R(asf=0.05, gep=15e16, Tdeb=200, Tc=65, Vat=1e-28, mu_at=4)
        R_FGT = SimAnalysis.get_R(asf=0.04, gep=1.33e18, Tdeb=190, Tc=220, Vat=1.7e-29, mu_at=2)
        R_CrI3 = SimAnalysis.get_R(asf=0.175, gep=4.05e16, Tdeb=134, Tc=61, Vat=1.35e-28, mu_at=3.87)

        # mag_av = np.sum(self.mags, axis=1) / len(self.mags[0])
        # te_av = np.sum(self.tes, axis=1) / len(self.mags[0])
        # start_plot = np.where(mag_av == np.amin(mag_av))[0][0]
        # te_rec = te_av[start_plot:]
        # mag_rec = mag_av[start_plot:]

        dm_dt_CGT = R_CGT*m*tem[:, np.newaxis]/65*(1-m/SimAnalysis.Brillouin(tem, m, 1.5, 65))
        dm_dt_FGT = R_FGT * m * tem[:, np.newaxis] / 220 * (1 - m / SimAnalysis.Brillouin(tem, m, 2, 220))
        dm_dt_CrI3 = R_CrI3 * m * tem[:, np.newaxis] / 61 * (1 - m / SimAnalysis.Brillouin(tem, m, 1.5, 61))

        tem_mesh, m_mesh = np.meshgrid(tem, m)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
        surf = ax.plot_surface(m_mesh, tem_mesh, dm_dt_CGT.T, cmap='inferno',
                               linewidth=0, antialiased=True, alpha=0.3)
        surf = ax.plot_surface(m_mesh, tem_mesh, dm_dt_FGT.T, cmap='Blues',
                               linewidth=0, antialiased=True, alpha=0.3)
        plt.colorbar(surf, label=r'dm/dt', shrink=0.5, aspect=10)

        # ax.plot(mag_rec, te_rec, color='black', lw=3.0)

        ax.set_xlabel(r'magnetization', fontsize=16)
        ax.set_ylabel(r'temperature', fontsize=16)
        ax.set_title(r'Map of Magnetization rate', fontsize=18)

        plt.show()
        return

    @staticmethod
    def get_R(asf, gep, Tdeb, Tc, Vat, mu_at):
        R = 8*asf*gep*Tc**2*Vat/mu_at/Tdeb**2/sp.k
        print('R = ', R*1e-12 , '1/ps')
        return 8*asf*gep*sp.k*Tc**2*Vat/mu_at/Tdeb**2

    @staticmethod
    def Brillouin(temps, mags, spin, Tc):
        pref_1 = (2*spin+1)/(2*spin)
        pref_2 = 1/(2*spin)
        x = 3*Tc*mags/temps[:, np.newaxis]*spin/(spin+1)

        term_1 = pref_1/np.tanh(pref_1*x)
        term_2 = -pref_2/np.tanh(pref_2*x)

        return term_1+term_2

    @staticmethod
    def create_mean_mag_map(Tc, S):
        # This function computes the mean field mean magnetization map by solving the self-consistent equation m=B(m, T)
        # As an output we get an interpolation function of the mean field magnetization at any temperature T<=T_c (this can of course be extended to T>T_c with zeros).

        # Start by defining a unity function m=m:
        def mag(m):
            return m

        # Define the Brillouin function as a function of scalars, as fsolve takes functions of scalars:
        def Brillouin(m, T):
            # This function takes input parameters
            #   (i) magnetization amplitude m_amp_grid (scalar)
            #   (ii) (electron) temperature (scalar)
            # As an output we get the Brillouin function evaluated at (i), (ii) (scalar)

            J = 3*S/(S+1)*sp.k*Tc
            eta = J * m / sp.k / T / Tc
            c1 = (2 * S + 1) / (2 * S)
            c2 = 1 / (2 * S)
            bri_func = c1 / np.tanh(c1 * eta) - c2 / np.tanh(c2 * eta)
            return bri_func

        # Then we also need a temperature grid. I'll make it course grained for low temperatures (<0.8*Tc) (small slope) and fine grained for large temperatures (large slope):
        temp_grid = np.array(list(np.arange(0, 0.8, 1e-3)) + list(np.arange(0.8, 1 + 1e-5, 1e-5)))

        # I will define the list of m_eq(T) here and append the solutions of m=B(m, T). It will have the length len(temp_grid) at the end.
        meq_list = [1.]

        # Define a function to find the intersection of m and B(m, T) for given T with scipy:
        def find_intersection_sp(m, Bm, m0):
            return scipy.optimize.fsolve(lambda x: m(x) - Bm(x), m0)

        # Find meq for every temperature, starting point for the search being (1-T/Tc)^(1/2), fill the list
        for i, T in enumerate(temp_grid[1:]):
            # Redefine the Brillouin function to set the temperature parameter (I did not find a more elegant solution to this):
            def Brillouin_2(m):
                return Brillouin(m, T)

            # Get meq:
            meq = find_intersection_sp(mag, Brillouin_2, np.sqrt(1 - T))
            if meq[0] < 0:  # This is a comletely unwarranted fix for values of meq<0 at temperatures very close to Tc, that fsolve produces. It seems to work though, as the interpolated function plotted by plot_mean_mags() seems clean.
                meq[0] *= -1
            # Append it to list me(T)
            meq_list.append(meq[0])
        meq_list[-1] = 0  # This fixes slight computational errors to fix m_eq(Tc)=0 (it produces something like m_eq[-1]=1e-7)
        return ip.interp1d(temp_grid, meq_list)

    @staticmethod
    def get_mean_mag_map(mmag_interpol, temp):
        mask = temp<1
        mmag_temp = np.zeros_like(temp)
        mmag_temp[mask] = mmag_interpol(temp[mask])
        return mmag_temp

    @staticmethod
    def get_umd_data(mat):

        assert mat == 'cri3' or 'cgt' or 'fgt', 'Choose cri3, cgt or fgt'

        if mat == 'cri3':
            data = np.loadtxt('ultrafast mag dynamics/CrI3_dat.txt')
            data[:, 1] += 1
        elif mat == 'cgt':
            data = np.loadtxt('ultrafast mag dynamics/CGT_dat.txt')
            data[:, 1] = -data[:, 1]
            data[:, 0] += 0.35
        elif mat == 'fgt':
            data = np.loadtxt('ultrafast mag dynamics/FGT_dat.txt')
            data[:, 1] = data[:, 1]/data[0, 1]
            data[:, 0] -= 0.15
        elif mat == 'cgt_thin':
            data = np.loadtxt('ultrafast mag dynamics/CGT_thin_dat.txt')
            data[:, 0] += 4
        elif mat == 'cgt_thick':
            data = np.loadtxt('ultrafast mag dynamics/CGT_thick_dat.txt')
            data[:, 0] += 4
        delay = data[:, 0]
        mag = data[:, 1]
        return delay, mag

    @staticmethod
    def fit_umd_data(mat, file):

        plt.figure(figsize=(9.7, 6.8))

        if mat == 'all':
            assert type(file) is list and len(file) == 3, 'Give three simulations as well'
            mats = ['cgt', 'fgt', 'cri3']
            for loop_mat, loop_file in zip(mats, file):
                exp_data = SimAnalysis.get_umd_data(loop_mat)
                sim_data = SimPlot(loop_file)
                delay, tes, tps, mags = sim_data.get_data()[:4]
                mags /= mags[0, 0]

                # plt.scatter(exp_data[0], exp_data[1])
                plt.plot(delay * 1e12, mags[:, 0] - 1, label=loop_mat)
                plt.legend(fontsize=14)

        elif mat == 'cgt_long':
            assert type(file) is list and len(file) == 2
            mats = ['cgt_thin', 'cgt_thick']
            labels = ['15 nm CGT', '150 nm CGT']
            colors = [(1, 150/255, 79/255), (0, 69/255, 126/255)]
            for loop_mat, loop_file, label, color in zip(mats, file, labels, colors):
                exp_data = SimAnalysis.get_umd_data(loop_mat)
                sim_data = SimPlot(loop_file)
                delay, tes, tps, mags = sim_data.get_data()[:4]

                if loop_mat == 'cgt_thin':
                    mags = SimAnalysis.get_kerr(mags=mags, pen_dep=2e-9, layer_thick=14e-9, norm=True)
                else:
                    mags = SimAnalysis.get_kerr(mags=mags, pen_dep=2e-9, layer_thick=1e-9, norm=True)

                plt.scatter(exp_data[0], exp_data[1], s=10.0, label=label, color=color)
                plt.plot(delay * 1e12, mags, lw=3.0, color=color)
                plt.legend(fontsize=18)

        else:
            exp_data = SimAnalysis.get_umd_data(mat)
            sim_data = SimPlot(file).get_data()[:4]
            delay, tes, tps, mags = sim_data[:4]
            mags /= mags[0, 0]

            def laser(sigma, amp, t, offset):
                amp = amp/np.sqrt(2*np.pi/sigma**2)
                gauss = np.exp(-t**2/2/sigma**2)
                return offset + amp * gauss

            plt.fill_between(delay*1e12, laser(60e-3, 2, delay*1e12, 0.93), np.ones_like(delay)*0.93, color=(159/255, 118/255, 85/255), label=r'pulse')

            plt.scatter(exp_data[0], exp_data[1], marker='x', label=r'Lichtenberg et al.', s=150, color=(185/255, 132/255, 140/255))
            plt.plot(delay*1e12, mags[:, 0], lw=3.0, label=r'simulation', color=(185/255, 132/255, 140/255))

        # plt.xlim(-1, 5)
        plt.legend(fontsize=20)
        plt.xlabel(r'delay [ps]', fontsize=24)
        plt.ylabel(r'norm. Kerr-signal', fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig('DPG/CGT_long_fit.pdf')

        plt.show()
        return

    @staticmethod
    def get_kerr(mags, pen_dep, layer_thick, norm):
        exp_decay = np.exp(-np.arange(len(mags.T)) * layer_thick / pen_dep)
        kerr_signal = np.sum(np.multiply(mags, exp_decay[np.newaxis, ...]), axis=1)/np.sum(exp_decay)
        if norm:
            kerr_signal = (kerr_signal - kerr_signal[0]) \
                               / np.abs(np.amin(kerr_signal - kerr_signal[0]))
        return kerr_signal

    @staticmethod
    def phonon_exp_fit(t, T0, Teq, exponent, delay):
        return (T0 - Teq) * np.exp(-1 / exponent * (t - delay)) + Teq

    @staticmethod
    def phonon_double_exp_fit(t, A, B, tau_p_1, tau_p_2, C):
        exp_1 = A * np.exp(-t/tau_p_1)
        exp_2 = B * np.exp(-t/tau_p_2)
        return exp_1 + exp_2 + C

    @staticmethod
    def fit_phonon_decay(file, first_layer_index, last_layer_index, start_time, end_time=None):

        # get the simulation data needed:
        sim_data = SimPlot(file).get_data()
        sim_delay = sim_data[0]
        sim_tp = sim_data[2]

        # if no end_time is chosen, set it at the last timestep:
        if end_time is None:
            end_time = sim_delay[-1]

        # find the indices corresponding to the chosen time intervall:
        first_time_index = finderb(start_time, sim_delay)[0]
        last_time_index = finderb(end_time, sim_delay)[0]

        # restrict the data to the time intervall:
        sim_delay = sim_delay[first_time_index: last_time_index]
        sim_tp = sim_tp[first_time_index: last_time_index, first_layer_index: last_layer_index+1]

        # average the phonon temperature dynamics:
        sim_tp_av = np.sum(sim_tp, axis=1)/(np.abs(first_layer_index-(last_layer_index+1)))

        dz = 2e-9
        pen_dep = 14e-9
        exp_decay = np.exp(-np.arange(len(sim_tp[0])) * dz / pen_dep)

        # sim_tp_av = np.sum(sim_tp * exp_decay, axis=1) / np.sum(exp_decay)
        tau_p_guess = 1e-9

        p0 = [sim_tp_av[-1], tau_p_guess]
        popt, cv = scipy.optimize.curve_fit(lambda t, Teq, exponent:SimAnalysis.phonon_exp_fit(t, sim_tp_av[0], Teq, exponent, sim_delay[0]), sim_delay, sim_tp_av, p0)

        while not np.all(np.isfinite(cv)):
            print('trying again')
            tau_p_guess *= 0.9
            p0 = [sim_tp_av[0], sim_tp_av[-1], tau_p_guess, sim_delay[0]]
            popt_single, cv_single = scipy.optimize.curve_fit(SimAnalysis.phonon_exp_fit, sim_delay, sim_tp_av, p0)
            popt, cv = popt_single, cv_single
        print('simulation file: ', file)
        print('T_eq, tau_p, = ', popt)
        print('standard deviation:', np.sqrt(np.diag(cv)))

        plt.plot(sim_delay, sim_tp_av, label=r'simulation')
        plt.plot(sim_delay, SimAnalysis.phonon_exp_fit(sim_delay, sim_tp_av[0], popt[0], popt[1], sim_delay[0]), label=r'fit')
        plt.legend(fontsize=14)
        plt.xlabel(r'delay [ns]')
        plt.ylabel(r'averaged phonon temperature [K]')
        plt.show()

        tau_p1_guess = 1e-10
        tau_p2_guess = 1e-9
        try:
            p0_double = [sim_tp_av[0] / 2, sim_tp_av[0] / 2, tau_p1_guess, tau_p2_guess, sim_tp_av[-1]]
            popt_double, cv_double = scipy.optimize.curve_fit(SimAnalysis.phonon_double_exp_fit, sim_delay, sim_tp_av, p0_double)
            popt, cv = popt_double, cv_double
            print('simulation file: ', file)
            print('amp_1, amp_2, tau_p_1, tau_p_2, T_eq = ', popt_double)
        except RuntimeError:
            # print(file)
            # plt.plot(sim_delay, sim_tp_av, label=r'simulation')
            # plt.show()
            tau_p1_guess *= 0.5
            tau_p2_guess *= 0.5
            p0_double = [sim_tp_av[0] / 2, sim_tp_av[0] / 2, tau_p1_guess, tau_p2_guess, sim_tp_av[-1]]
            popt_double, cv_double = scipy.optimize.curve_fit(SimAnalysis.phonon_double_exp_fit, sim_delay, sim_tp_av, p0_double)
            popt, cv = popt_double, cv_double

        # plt.plot(sim_delay, sim_tp_av, label=r'simulation')
        # plt.plot(sim_delay, SimAnalysis.phonon_double_exp_fit(sim_delay, *popt_double), label=r'fit')
        # plt.legend(fontsize=14)
        # plt.xlabel(r'delay [ns]')
        # plt.ylabel(r'averaged phonon temperature [K]')
        # plt.show()

        return popt, cv

    @staticmethod
    def fit_mag_decay(file, t1, t_max=None):
        # This method fits the average of a simulation set of magnetization dynamics on long timescales after
        # e-p-equilibration to a function:
        # f(t) = const. + a*exp(-t/tau_1)+b*tanh((t-t_1)/tau_2)

        # Input:
        # file (string). Path of the simulation folder from Results
        # t1 (float). Time in s of when the second slow demagnetization sets in (around e-p-equilibration time)

        # Returns:
        # popt_all, cv_all (lists). Optimized parameters of the fit and corresponding error values in the order as
        # in exp_tanh(t, *args)

        # get the data to fit:
        sim_data = SimPlot(file).get_data()
        sim_delay = sim_data[0]
        sim_mags = sim_data[3]

        # average the magnetization:
        mag_av = np.sum(sim_mags, axis=1)/len(sim_mags[0])

        # sort out the time intervals
        first_time_index = finderb(t1, sim_delay)[0]
        if t_max is None:
            last_time_index = len(sim_delay) + 1
        else:
            last_time_index = finderb(t_max, sim_delay)[0]

        mmin_time_index = finder_nosort(np.amin(mag_av), mag_av)[0]
        delay_phase_1 = sim_delay[first_time_index:mmin_time_index]
        mag_phase_1 = mag_av[first_time_index:mmin_time_index]

        tau_1_try = 5e-10
        p0_1 = [mag_av[mmin_time_index+first_time_index], tau_1_try, 2e-12]
        def LLB_demag(t, mmag, tau_1, t_offset):
            denom = np.sqrt(1-(1-mmag**2)*np.exp(-2*(t-t_offset)/tau_1))
            return mmag/denom

        popt_1, cv_1 = scipy.optimize.curve_fit(LLB_demag, delay_phase_1, mag_phase_1, p0_1)
        while not np.all(np.isfinite(cv_1)):
            tau_1_try *= 0.9
            p0_1 = [mag_av[mmin_time_index + first_time_index], tau_1_try, 2e-12]
            popt_1, cv_1 = scipy.optimize.curve_fit(LLB_demag, delay_phase_1, mag_phase_1, p0_1)

        print('simulation file: ', file)
        print('fit_mmag, fit_tau_1, fit_t_offset = ', popt_1)

        plt.plot(delay_phase_1, mag_phase_1, label='sim', ls='dotted', color='orange')
        plt.plot(delay_phase_1, LLB_demag(delay_phase_1, *popt_1), label='LLB_fit')

        plt.legend(fontsize=14)
        plt.xlabel(r'delay [s]', fontsize=16)
        plt.ylabel(r'average magnetization', fontsize=16)
        plt.show()

        return popt_1, cv_1

    @staticmethod
    def fit_dm_dt(file, t1, t_max=None, p0_initial=[0.4, 0.5, 4e-12, 5e-11, 0.3, 1e-9, 5e-10]):

        sim_data = SimPlot(file).get_data()
        sim_delay = sim_data[0]
        sim_mags = sim_data[3]

        # average the magnetization:
        mag_av = np.sum(sim_mags, axis=1) / len(sim_mags[0])

        mag_diff = np.diff(mag_av)
        time_diff = np.diff(sim_delay)
        dm_dt = mag_diff / time_diff

        start_index = finderb(t1, sim_delay[1:])[0]
        if t_max is None:
            last_time_index = len(sim_delay) + 1
        else:
            last_time_index = finderb(t_max, sim_delay)[0]

        delay_res = sim_delay[1+start_index:last_time_index]
        dm_dt_res = dm_dt[start_index:last_time_index]

        def dm_dt_fit_func(t, exp_scale, exp_offset, tau_1, tanh_scale, tanh_offset, tau_2):
            exp_term = -exp_scale/tau_1*np.exp(-(t-exp_offset)/tau_1)
            tanh_term = tanh_scale/tau_2 * (1-np.tanh((t-tanh_offset)/tau_2)**2)
            func = exp_term + tanh_term
            return func

        p0 = p0_initial

        # popt, cv = scipy.optimize.curve_fit(dm_dt_fit_func, delay_res, dm_dt_res, p0)

        # print('exp_scale, exp_offset, tau_1 = ', popt[:3])
        # print('tanh_scale, tanh_offset, tau_2 = ', popt[3:])

        plt.plot(delay_res, dm_dt_res, ls='dotted', color='orange', label='sim')
        # plt.plot(delay_res, dm_dt_fit_func(delay_res, *popt), color='blue', label='fit')
        plt.plot(delay_res, dm_dt_fit_func(delay_res, *p0), label='default')

        plt.legend(fontsize=14)
        plt.xlabel(r'delay [s]', fontsize=16)
        plt.ylabel(r'dm/dt [1/s]', fontsize=16)
        plt.show()

        return popt, cv

    @staticmethod
    def fit_FGT_remag(file, s, tc):
        sim_data = SimPlot(file).get_data()
        sim_delay = sim_data[0]
        sim_mags = sim_data[3]
        sim_tps = sim_data[2][:, :8]

        mag_av = np.sum(sim_mags, axis=1) / len(sim_mags[0])

        mmin_time_index = finder_nosort(np.amin(mag_av), mag_av)[0]
        delay_phase_1 = sim_delay[mmin_time_index:]
        mag_phase_1 = mag_av[mmin_time_index:]
        tp_phase_1 = np.sum(sim_tps[mmin_time_index:], axis=1)/len(sim_tps.T)

        # popt, cv_opt = SimAnalysis.fit_phonon_decay(file=file, first_layer_index=7, last_layer_index=13, start_time=sim_delay[mmin_time_index], end_time=None)
        mmag_func = SimAnalysis.create_mean_mag_map(tc, s)
        # phonon_temp = SimAnalysis.phonon_exp_fit(delay_phase_1, *popt)
        # # mmag_T = mmag_func(phonon_temp/220.)
        mmag_T = SimAnalysis.get_mean_mag_map(mmag_func, tp_phase_1/tc)
        #
        # T = np.linspace(0, 0.999, 100)
        # plt.plot(T, mmag_func(T), label=r'mmag (S=1.5)')
        # plt.plot(T, (1-T)**(0.28), label=r'power law')
        # plt.legend()
        # plt.xlabel(r'$T/T_c$')
        # plt.ylabel(r'm_eq')
        # plt.show()

        plt.plot(delay_phase_1*1e9, mmag_T, label=r'$m_{eq}$(T(t))', ls=':', lw=5.0)
        plt.plot(delay_phase_1*1e9, mag_phase_1, label=r'our model')
        # plt.plot(delay_phase_1, (1-tp_phase_1/220.)**(0.28), ls=':', lw=5.0, label=r'power law')
        plt.xlabel(r'delay [ns]', fontsize=16)
        plt.ylabel(r'magnetization', fontsize=16)
        plt.legend(fontsize=16)
        # plt.ylim(0.972, 1.001)
        plt.savefig('cgt_sio2_remag.pdf')
        plt.show()


        def fit_m_diff(t, tau_1, tau_2, A, B):
            return -A * np.exp((-t-5e-10)/tau_1) + B * np.exp((-t-5e-10)/tau_2)

        p0_diff = [5e-10, 6.7e-10, 2, 1.5]
        # popt, cv = scipy.optimize.curve_fit(fit_m_diff, delay_phase_1 * 1e9, (mmag_T - mag_phase_1))

        plt.plot(delay_phase_1, (mmag_T**2-mag_phase_1**2), label=r'difference')
        # plt.plot(delay_phase_1, fit_m_diff(delay_phase_1, *popt), label=r'fit')
        plt.plot(delay_phase_1, fit_m_diff(delay_phase_1, *p0_diff), label=r'default params')
        plt.legend()
        plt.show()
        return

    def fit_all_phonon(self, save_file_name, first_layer, last_layer, last_time):

        # create an array to store all fit parameters: ferro X sub X fluence X param
        all_phonon_params = np.zeros((3, 9, 4, 5), dtype=float)
        # determine where to save:
        for file in self.files:

            if 'FGT' in file:
                i = 0
                start_time = 10e-12
            elif 'CGT' in file:
                i = 1
                start_time = 10e-12
            elif 'CrI3' in file:
                i = 2
                start_time = 10e-12
            else:
                print('Did not find the ferromagnet you made me search')

            if 'AIN' in file:
                j = 0
            elif 'graphene' in file:
                j = 1
            elif 'MoS2' in file:
                j = 2
            elif 'hBN' in file:
                j = 3
            elif 'Bi2Te3' in file:
                j = 4
            elif 'WS2' in file:
                j = 5
            elif 'Al2O3' in file:
                j = 6
            elif 'SiO2' in file:
                j = 7
            elif 'WSe2' in file:
                j = 8
            else:
                print('Did not find the substrate you made me search')

            if 'flu_0.3' in file:
                k = 0
            elif 'flu_0.5' in file:
                k = 1
            elif 'flu_1.0' in file:
                k = 2
            elif 'flu_1.5' in file:
                k = 3
            else:
                print('Did not find the fluence you made me search')

            popt, cv = SimAnalysis.fit_phonon_decay(file, first_layer, last_layer, start_time, last_time)
            # popt = Teq, exponent, delay
            all_phonon_params[i, j, k] = popt
            np.save('Results/' + save_file_name + '.npy', all_phonon_params)

        return

    def fit_all_mag(self, save_file_name):
        # create an array to store all fit parameters: ferro X sub X fluence X param
        all_mag_params = np.zeros((3, 5, 4, 3), dtype=float)
        # determine where to save:
        for file in self.files:

            if 'FGT' in file:
                i = 0
                start_time = 5e-12
            elif 'CGT' in file:
                i = 1
                start_time = 5e-12
            elif 'CrI3' in file:
                i = 2
                start_time = 5e-12
            else:
                print('Did not find the ferromagnet you made me search')

            if 'AIN' in file:
                j = 0
            elif 'graphene' in file:
                j = 1
            elif 'MoS2' in file:
                j = 2
            elif 'hBN' in file:
                j = 3
            elif 'BiTe' in file:
                j = 4
            elif 'WS2' in file:
                j = 5
            elif 'Al2O3' in file:
                j = 6
            elif 'SiO2' in file:
                j = 7
            elif 'WSe2' in file:
                j = 8
            else:
                print('Did not find the substrate you made me search')

            if 'flu_0.3' in file:
                k = 0
            elif 'flu_0.5' in file:
                k = 1
            elif 'flu_1.0' in file:
                k = 2
            elif 'flu_1.5' in file:
                k = 3
            else:
                print('Did not find the fluence you made me search')

            popt, cv = SimAnalysis.fit_mag_decay(file, start_time)
            # popt = T0, Teq, exponent, delay
            all_mag_params[i, j, k] = popt

        np.save('Results/' + save_file_name + '.npy', all_mag_params)
        return

    @staticmethod
    def plot_phonon_params(file, savefile):

        # load the array:
        all_phonon_params = np.load('Results/' + file)

        # constant fluence, all substrates, each material:
        tau_p1_params = [[[], []], [[], []], [[], []]]
        tau_p2_params = [[[], []], [[], []], [[], []]]
        teq_params = [[[], []], [[], []], [[], []]]

        for i, ferro_row in enumerate(all_phonon_params):
            if i == 0:
                kappa_ferro = 0.5
            if i == 1:
                kappa_ferro = 1.
            if i == 2:
                kappa_ferro = 1.36
            for j, sub_row in enumerate(ferro_row):
                if j == 0:
                    kappa = 8.5
                elif j == 1:
                    kappa = 6
                elif j == 2:
                    kappa = 5
                elif j == 3:
                    kappa = 5
                elif j == 4:
                    kappa = 1.8
                elif j == 5:
                    kappa = 1.7
                elif j == 6:
                    kappa = 1.
                elif j == 7:
                    kappa = 1.
                elif j == 8:
                    kappa = 0.35
                for k, param in enumerate(sub_row[1, 2:]):
                    if k == 2:
                        teq_params[i][0].append(kappa)
                        teq_params[i][1].append(param)
                    elif k == 0:
                        tau_p1_params[i][0].append(kappa)
                        if sub_row[1, 0] *100 < sub_row[1, 1]:
                            tau_p1_params[i][1].append(sub_row[1, 3]*1e9)
                        else:
                            tau_p1_params[i][1].append(param*1e9)
                    elif k == 1:
                        tau_p2_params[i][0].append(kappa)
                        tau_p2_params[i][1].append(param*1e9)

        teq_params = np.array(teq_params)
        tau_p1_params = np.array(tau_p1_params)
        tau_p2_params = np.array(tau_p2_params)
        # create figure:
        fig = plt.figure(layout='constrained', figsize=(7, 9))
        fig_teq = plt.subplot(3, 1, 1)
        fig_tau_p1 = plt.subplot(3, 1, 2)
        fig_tau_p2 = plt.subplot(3, 1, 3)

        # arange figure:
        fig_tau_p2.set_xlabel(r'substrate $\kappa$ [W/mK]', fontsize=20)

        fig_teq.set_ylabel(r'$T_{\rm{eq}}$ [K]', fontsize=20)
        fig_tau_p1.set_ylabel(r'$\tau_{p1}$ [ns]', fontsize=20)
        fig_tau_p2.set_ylabel(r'$\tau_{p2}$ [ns]', fontsize=20)

        fig_tau_p2.tick_params(axis='x', labelsize=18)
        fig_teq.xaxis.set_tick_params(labelbottom=False)
        fig_tau_p1.xaxis.set_tick_params(labelbottom=False)

        fig_teq.tick_params(axis='y', labelsize=18)
        fig_tau_p1.tick_params(axis='y', labelsize=18)
        fig_tau_p2.tick_params(axis='y', labelsize=18)

        colors = [np.array([185, 132, 140])/255, np.array([47, 112, 175])/255, np.array([128, 100, 145])/255]
        labels = [r'FGT', r'CGT', r'CrI3']

        for i in range(len(colors)):
            fig_teq.plot(teq_params[i][0], teq_params[i][1], ls='dashed', marker='o', lw=0.5, color=colors[i], label=labels[i])
            fig_tau_p1.plot(tau_p1_params[i][0], (tau_p1_params[i][1]), ls='dashed', marker='o', lw=0.5, color=colors[i])
            fig_tau_p2.plot(tau_p2_params[i][0], tau_p2_params[i][1], ls='dashed', marker='o', lw=0.5, color=colors[i])

        fig_teq.legend(fontsize=18)
        plt.savefig('Results/' + savefile + '.pdf')

        plt.show()
        return

















