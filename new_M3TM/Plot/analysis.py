import numpy as np
import scipy
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

    def mag_tem_plot(self):
        for file in self.files:
            delays, mags, tes, tps = self.get_data(file)

            mag_av = np.sum(mags, axis=1)/len(mags[0])
            te_av = np.sum(tes, axis=1)/len(mags[0])

            start_plot = np.where(mag_av == np.amin(mag_av))[0][0]

            def mag(tem, tc):
                return np.real(np.sqrt(1-tem/tc))

            te_rec = te_av[start_plot:]
            mag_rec = mag_av[start_plot:]

            plt.plot(te_rec, mag_rec, label=str(file))

            # plt.plot(te_rec, mag(te_rec, 65), label=r'$m(T)=\sqrt{1-\frac{T}{T_c}}$')
        plt.xlabel('temperature', fontsize=16)
        plt.ylabel('magnetization', fontsize=16)
        plt.title('Magnetization over Temperature', fontsize=18)
        plt.legend(fontsize=14)

        plt.xscale('linear')
        plt.gca().invert_xaxis()
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
    def get_dm_dt(R, S, tc, tem, mag):
        J = 3 * sp.k * tc * S / (S + 1)
        arbsc = R / tc ** 2 / sp.k
        ms = (np.arange(2 * S + 1) + np.array([-S for i in range(int(2 * S) + 1)]))
        s_up_eig_squared = -np.power(ms, 2) - ms + S ** 2 + S
        s_dn_eig_squared = -np.power(ms, 2) + ms + S ** 2 + S

        h_mf = np.multiply(j_sam, mag)
        eta = np.divide(h_mf, np.multiply(2 * spin_sam * sp.k, te[el_mag_mask])).astype(float)
        incr_pref = arbsc_sam*tp*h_mf/4/spin_sam/np.sinh(eta)

        fs_up = np.multiply(s_up_eig_sq_sam, fs)
        fs_dn = np.multiply(s_dn_eig_sq_sam, fs)
        rate_up_fac = incr_pref * np.exp(-eta)
        rate_dn_fac = incr_pref * np.exp(eta)

        rate_up_loss = np.multiply(rate_up_fac[..., np.newaxis], fs_up)
        rate_dn_loss = np.multiply(rate_dn_fac[..., np.newaxis], fs_dn)

        rate_up_gain = np.roll(rate_up_loss, 1)
        rate_dn_gain = np.roll(rate_dn_loss, -1)

        dfs_dt = rate_up_gain + rate_dn_gain - rate_up_loss - rate_dn_loss

    @staticmethod
    def get_R(asf, gep, Tdeb, Tc, Vat, mu_at):
        R = 8*asf*gep/sp.k*Tc**2*Vat/mu_at/Tdeb**2
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

    def create_mean_mag_map(self):
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

            eta = self.J * m / sp.k / T / self.Tc
            c1 = (2 * self.S + 1) / (2 * self.S)
            c2 = 1 / (2 * self.S)
            bri_func = c1 / np.tanh(c1 * eta) - c2 / np.tanh(c2 * eta)
            return bri_func

        # Then we also need a temperature grid. I'll make it course grained for low temperatures (<0.8*Tc) (small slope) and fine grained for large temperatures (large slope):
        temp_grid = np.array(list(np.arange(0, 0.8, 1e-3)) + list(np.arange(0.8, 1 + 1e-5, 1e-5)))

        # I will define the list of m_eq(T) here and append the solutions of m=B(m, T). It will have the length len(temp_grid) at the end.
        meq_list = [1.]

        # Define a function to find the intersection of m and B(m, T) for given T with scipy:
        def find_intersection_sp(m, Bm, m0):
            return ip.fsolve(lambda x: m(x) - Bm(x), m0)

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
        meq_list[
            -1] = 0  # This fixes slight computational errors to fix m_eq(Tc)=0 (it produces something like m_eq[-1]=1e-7)
        return ip.interp1d(temp_grid, meq_list)

    @staticmethod
    def get_umd_data(mat):

        assert mat == 'cri3' or 'cgt' or 'fgt', 'Choose cri3, cgt or fgt'

        if mat == 'cri3':
            data = np.loadtxt('ultrafast mag dynamics/CrI3_dat.txt')
        elif mat == 'cgt':
            data = np.loadtxt('ultrafast mag dynamics/CGT_dat.txt')
            data[:, 1] = -data[:, 1] - 1
            data[:, 0] += 0.35
        elif mat == 'fgt':
            data = np.loadtxt('ultrafast mag dynamics/FGT_dat.txt')
            data[:, 1] = data[:, 1]/data[0, 1] - 1
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

        plt.figure(figsize=(8, 6))

        if mat == 'all':
            assert type(file) is list and len(file) == 3, 'Give three simulations as well'
            mats = ['cgt', 'fgt', 'cri3']
            for loop_mat, loop_file in zip(mats, file):
                exp_data = SimAnalysis.get_umd_data(loop_mat)
                sim_data = SimPlot(loop_file)
                delay, tes, tps, mags = sim_data.get_data()[:4]
                mags /= mags[0, 0]

                plt.scatter(exp_data[0], exp_data[1])
                plt.plot(delay * 1e12, mags[:, 0] - 1, label=loop_mat)
                plt.legend(fontsize=14)

        elif mat == 'cgt_long':
            assert type(file) is list and len(file) == 2
            mats = ['cgt_thin', 'cgt_thick']
            for loop_mat, loop_file in zip(mats, file):
                exp_data = SimAnalysis.get_umd_data(loop_mat)
                sim_data = SimPlot(loop_file)
                delay, tes, tps, mags = sim_data.get_data()[:4]
                if loop_mat == 'cgt_thin':
                    mags = SimAnalysis.get_kerr(mags=mags, pen_dep=1e-9, layer_thick=2e-9, norm=True)
                else:
                    mags = SimAnalysis.get_kerr(mags=mags, pen_dep=1e-9, layer_thick=2e-9, norm=True)

                plt.scatter(exp_data[0], exp_data[1], s=4.0)
                plt.plot(delay * 1e12, mags, label=loop_mat, lw=2.0)
                plt.legend(fontsize=14)

        else:
            exp_data = SimAnalysis.get_umd_data(mat)
            sim_data = SimPlot(file).get_data()[:4]
            delay, tes, tps, mags = sim_data.get_data()[:4]
            mags /= mags[0, 0]

            plt.scatter(exp_data[0], exp_data[1])
            plt.plot(delay*1e12, mags[:, 0]-1)

        # plt.xlim(-1, 25)

        plt.xlabel(r'delay [ps]', fontsize=16)
        plt.ylabel(r'Kerr signal', fontsize=16)

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

        # define the fit function:
        def phonon_exp_fit(t, T0, Teq, exponent):
            return (T0-Teq)*np.exp(-1/exponent*(t-sim_delay[0]))+Teq

        p0 = [sim_tp_av[0], sim_tp_av[0]/5, 1e-9]
        popt, cv = scipy.optimize.curve_fit(phonon_exp_fit, sim_delay, sim_tp_av, p0)

        print('T_0, T_eq, exponent [s] = ', popt)
        print('standard deviation:', np.sqrt(np.diag(cv)))

        plt.plot(sim_delay, sim_tp_av, label=r'simulation')
        plt.plot(sim_delay, phonon_exp_fit(sim_delay, *popt), label=r'fit')

        plt.legend(fontsize=14)
        plt.xlabel(r'delay [ps]')
        plt.ylabel(r'averaged phonon temperature [K]')
        plt.show()

        return popt, cv

    @staticmethod
    def fit_mag_data(file, t1):
        # This method fits the average of a simulation set of magnetization dynamics on long timescales after
        # e-p-equilibration to a function:
        # f(t) = const. + a*exp(-t/tau_1)+b*tanh((t-t_1)/tau_2)

        # Input:
        # file (string). Path of the simulation folder from Results
        # t1 (float). Time in s of when the second slow demagnetization sets in (around e-p-equilibration time)

        # Returns:
        # popt_1, cv_1, popt_2, cv_2 (lists). Optimized parameters of the fit and corresponding error values
        # for term 1 and 2 of the fit function.

        # get the data to fit:
        sim_data = SimPlot(file).get_data()
        sim_delay = sim_data[0]
        sim_mags = sim_data[3]

        # average the magnetization:
        mag_av = np.sum(sim_mags, axis=1)/len(sim_mags[0])

        # sort out the time intervals
        first_time_index = finderb(t1, sim_delay)[0]
        second_time_index = finder_nosort(np.amin(mag_av), mag_av)[0]

        # restrict the data to the time intervals:
        delay_phase_1 = sim_delay[first_time_index: second_time_index]
        mag_phase_1 = mag_av[first_time_index: second_time_index]

        delay_phase_2 = sim_delay[second_time_index:]
        mag_phase_2 = mag_av[second_time_index:]

        delay_phase_12 = sim_delay[first_time_index:]
        mag_phase_12 = mag_av[first_time_index:]

        # define fit functions:
        def exp_phase_1(t, m0, m_min, a):
            return (m0-m_min) * np.exp(-t/a) + m_min

        def exp_phase_2(t, t_mid, m_min, m_inf, a):
            return (m_inf-m_min) * np.tanh((t-t_mid)/a) + m_inf

        def exp_tanh(t, offset, exp_scale, exp_offset, tau_1, tanh_scale, tanh_offset, tau_2):

            exp_part = exp_scale*np.exp(-(t-exp_offset)/tau_1)
            tanh_part = tanh_scale*np.tanh((t-tanh_offset)/tau_2)
            full_func = offset + exp_part + tanh_part

            return full_func

        p0_1 = [0.8, 0.1, 1e-10]
        p0_2 = [1e-9, 0.1, 1, 1e-9]
        p0_all = [0.5, 0.3, 4e-12 ,1e-10, 0.3, 1e-9, 5e-10]

        # popt_1, cv_1 = scipy.optimize.curve_fit(exp_phase_1, delay_phase_1, mag_phase_1, p0_1)
        # popt_2, cv_2 = scipy.optimize.curve_fit(exp_phase_2, delay_phase_2, mag_phase_2, p0_2)
        popt_all, cv_all = scipy.optimize.curve_fit(exp_tanh, delay_phase_12, mag_phase_12, p0_all)

        # print('m_0, m_min, tau_1 = ', popt_1)
        # print('t_mid, m_inf, tau_2 = ', popt_2[0], popt_2[2], popt_2[3])
        print('offset, exp_scale, exp_offset, tau_1 = ', popt_all[:4])
        print('tanh_scale, tanh_offset, tau_2 = ', popt_all[4:])

        plt.plot(sim_delay, mag_av, label='sim', ls='dotted', color='orange')
        # plt.plot(delay_phase_1, exp_phase_1(delay_phase_1, *popt_1), color='pink', label='exp')
        # plt.plot(delay_phase_2, exp_phase_2(delay_phase_2, *popt_2), color='purple', label='tanh')
        plt.plot(delay_phase_12, exp_tanh(delay_phase_12, *popt_all), color='blue', label='combined')


        plt.legend(fontsize=14)
        plt.xlabel(r'delay [s]', fontsize=16)
        plt.ylabel(r'average magnetization', fontsize=16)
        plt.show()

        return popt_1, cv_1, popt_2, cv_2
