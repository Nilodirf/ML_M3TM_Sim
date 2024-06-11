import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import constants as sp
from scipy import interpolate as ip
from scipy import optimize as op
from matplotlib import colors as mplcol

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

    @staticmethod
    def plot_dmdt():
        m = np.linspace(0, 1, num=100)
        tem = np.arange(0, 70)

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
        norm = mplcol.Normalize(vmin=-9e-37, vmax=9e-37)
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
        fig, ax = plt.subplots(figsize=(8, 6))
        # surf = ax.plot_surface(m_mesh, tem_mesh, dm_dt_CGT.T, cmap='bwr',
        #                        linewidth=0, antialiased=True, alpha=0.8, vmin=-9e-37, vmax=9e-37)
        surf = ax.pcolormesh(m_mesh, tem_mesh, dm_dt_CGT.T, antialiased=True, cmap='bwr', shading='nearest', norm=norm)
        # surf = ax.plot_surface(m_mesh, tem_mesh, dm_dt_FGT.T, cmap='Blues',
        #                        linewidth=0, antialiased=True, alpha=0.3)
        plt.colorbar(surf, label=r'dm/dt', shrink=0.5, aspect=10)

        # ax.plot(mag_rec, te_rec, color='black', lw=3.0)

        ax.set_xlabel(r'Magnetization', fontsize=16)
        ax.set_ylabel(r'Temperature', fontsize=16)
        ax.set_title(r'Map of magnetization rate', fontsize=18)
        ax.hlines(65, 0, 1, lw=1.0, color='black', ls='dashed')
        # ax.view_init(5, 30)
        ax.set_xlim(0,1)
        ax.set_ylim(0, 69)
        plt.savefig('Results/CGT Paper/dm_dt.pdf')
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

    @staticmethod
    def create_mean_mag_map(S, Tc):
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
            return op.fsolve(lambda x: m(x) - Bm(x), m0)

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
        temp_grid = np.append(temp_grid, 10.)
        meq_list.append(0.)
        return ip.interp1d(temp_grid, meq_list)


    @staticmethod
    def show_mean_field_mag(S, Tc, savepath):
        temps = np.linspace(0,1.5, 150)
        m_eq = SimAnalysis.create_mean_mag_map(S, Tc)

        plt.figure(figsize=(8, 6))
        plt.plot(temps, m_eq(temps), lw=2.0)
        plt.xlabel(r'T/T_C', fontsize=16)
        plt.ylabel(r'm$_{\rm{eq}}$', fontsize=16)
        plt.title(r'Mean field magnetization curve', fontsize=18)
        plt.xlim(0, 1.5)
        plt.ylim(-0.01, 1.01)
        plt.savefig(savepath)
        plt.show()

        return

    @staticmethod
    def plot_review_answer():

        meq = SimAnalysis.create_mean_mag_map(S=1.5, Tc=65.)

        t_15nm = np.load('Results/CGT Paper/15nm_fl_0.5_pristine/delay.npy')
        te_15nm_raw = np.load('Results/CGT Paper/15nm_fl_0.5_pristine/tes.npy')
        mag_15nm_raw = np.load('Results/CGT Paper/15nm_fl_0.5_pristine/ms.npy')
        tp_15nm_raw = np.load('Results/CGT Paper/15nm_fl_0.5_pristine/tps.npy')
        te_15nm = np.sum(te_15nm_raw, axis=1)/7
        mag_15nm = np.sum(mag_15nm_raw, axis=1)/7
        tp_15nm = np.sum(tp_15nm_raw[:, 7:14], axis=1)/7
        meq_15nm = meq(te_15nm/65.)
        t_TC_15nm = finderb(0.6476, t_15nm*1e9)[0]

        t_90nm = np.load('Results/CGT Paper/90nm_fl_0.5_pristine/delay.npy')
        te_90nm_raw = np.load('Results/CGT Paper/90nm_fl_0.5_pristine/tes.npy')
        mag_90nm_raw = np.load('Results/CGT Paper/90nm_fl_0.5_pristine/ms.npy')
        tp_90nm_raw = np.load('Results/CGT Paper/90nm_fl_0.5_pristine/tps.npy')
        te_90nm = np.sum(te_90nm_raw, axis=1)/45
        mag_90nm = np.sum(mag_90nm_raw, axis=1)/45
        tp_90nm = np.sum(tp_90nm_raw[:, 7:52], axis=1)/45
        meq_90nm = meq(te_90nm/65.)
        t_TC_90nm = finderb(1.2084, t_90nm*1e9)[0]

        fig, axs = plt.subplots(3, 2, sharex='col', figsize=(8, 6), width_ratios=[0.6476, 5], layout='compressed')
        fig.suptitle(r'15 nm CGT', fontsize=18)

        axs[0][0].plot(t_15nm[:t_TC_15nm]*1e9, mag_15nm[:t_TC_15nm], lw=2.0, label='m(t)')
        axs[0][0].plot(t_15nm[:t_TC_15nm]*1e9, meq_15nm[:t_TC_15nm], lw=2.0, label=r'm$_{\rm{eq}}$(Te(t))')
        axs[0][0].set_ylabel(r'magnetization', fontsize=16)
        axs[0][0].set_xlim(-0.1, 0.6476)
        # axs[0][0].legend(fontsize=14, loc='center right')
        axs[0][0].set_ylim(-0.01, 1.01)

        axs[0][1].plot(t_15nm[t_TC_15nm:]*1e9, mag_15nm[t_TC_15nm:], lw=2.0, label='m(t)')
        axs[0][1].plot(t_15nm[t_TC_15nm:]*1e9, meq_15nm[t_TC_15nm:], lw=2.0, label=r'm$_{\rm{eq}}$(Te(t))')
        # axs[0][1].set_ylabel(r'av. magnetization', fontsize=16)
        axs[0][1].yaxis.tick_right()
        axs[0][1].set_xlim(0.6476, 5)
        axs[0][1].legend(fontsize=14, loc='center right')
        axs[0][1].set_ylim(-0.01, 1.01)

        axs[1][0].plot(t_15nm[:t_TC_15nm]*1e9, -(mag_15nm-meq_15nm)[:t_TC_15nm], lw=2.0, color='purple')
        axs[1][0].hlines(y=0, xmin=-0.1, xmax=5, color='black', ls='dashed', alpha=0.8)
        axs[1][0].set_ylabel(r'm$_{\rm{eq}}$(t)-m(t)', fontsize=16)

        axs[1][1].plot(t_15nm[t_TC_15nm:]*1e9, -(mag_15nm-meq_15nm)[t_TC_15nm:], lw=2.0, color='purple')
        axs[1][1].hlines(y=0, xmin=-0.1, xmax=5, color='black', ls='dashed', alpha=0.8)
        axs[1][1].yaxis.tick_right()
        # axs[1][1].set_ylabel(r'm$_{eq}$(t)-m(t)', fontsize=16)

        axs[2][0].plot(t_15nm[:t_TC_15nm]*1e9, te_15nm[:t_TC_15nm], lw=2.0, label=r'T$_e$', color='red')
        axs[2][0].plot(t_15nm[:t_TC_15nm]*1e9, tp_15nm[:t_TC_15nm], lw=2.0, label=r'T$_p$', color='darkgreen')
        axs[2][0].set_ylabel(r'Temperature [K]', fontsize=16)
        # axs[2][0].set_xlabel(r'delay [ns]', fontsize=16)
        axs[2][0].hlines(y=65, xmin=-0.1, xmax=5, lw=1.5, color='black', ls='solid', label=r'T$_C$')
        # axs[2][0].legend(fontsize=14, loc='center right')

        axs[2][1].plot(t_15nm[t_TC_15nm:]*1e9, te_15nm[t_TC_15nm:], lw=2.0, label=r'T$_e$', color='red')
        axs[2][1].plot(t_15nm[t_TC_15nm:]*1e9, tp_15nm[t_TC_15nm:], lw=2.0, label=r'T$_p$', color='darkgreen')
        # axs[2][1].set_ylabel(r'Temperature [K]', fontsize=16)
        axs[2][1].yaxis.tick_right()
        axs[2][1].set_xlabel(r'delay [ns]', fontsize=16)
        axs[2][1].hlines(y=65, xmin=-0.1, xmax=5, lw=1.5, color='black', ls='solid', label=r'T$_C$')
        axs[2][1].legend(fontsize=14, loc='center right')
        plt.savefig('Results/CGT Paper/15nm_review_reply.pdf')
        plt.show()

        fig, axs = plt.subplots(3, 2, sharex='col', figsize=(8, 6), width_ratios=[1.2084, 5], layout='compressed')
        fig.suptitle(r'90 nm CGT', fontsize=18)

        axs[0][0].plot(t_90nm[:t_TC_90nm]*1e9, mag_90nm[:t_TC_90nm], lw=2.0, label='m(t)')
        axs[0][0].plot(t_90nm[:t_TC_90nm]*1e9, meq_90nm[:t_TC_90nm], lw=2.0, label=r'm$_{\rm{eq}}$(Te(t))')
        axs[0][0].set_ylabel(r'magnetization', fontsize=16)
        axs[0][0].set_xlim(-0.1, 1.2084)
        # axs[0][0].legend(fontsize=14, loc='center right')
        axs[0][0].set_ylim(-0.01, 1.01)

        axs[0][1].plot(t_90nm[t_TC_90nm:]*1e9, mag_90nm[t_TC_90nm:], lw=2.0, label='m(t)')
        axs[0][1].plot(t_90nm[t_TC_90nm:]*1e9, meq_90nm[t_TC_90nm:], lw=2.0, label=r'm$_{\rm{eq}}$(Te(t))')
        # axs[0][1].set_ylabel(r'av. magnetization', fontsize=16)
        axs[0][1].yaxis.tick_right()
        axs[0][1].set_xlim(1.2084, 5)
        axs[0][1].legend(fontsize=14, loc='upper right')
        axs[0][1].set_ylim(-0.01, 1.01)

        axs[1][0].plot(t_90nm[:t_TC_90nm]*1e9, -(mag_90nm-meq_90nm)[:t_TC_90nm], lw=2.0, color='purple')
        axs[1][0].hlines(y=0, xmin=-0.1, xmax=5, color='black', ls='dashed', alpha=0.8)
        axs[1][0].set_ylabel(r'm$_{\rm{eq}}$(t)-m(t)', fontsize=16)

        axs[1][1].plot(t_90nm[t_TC_90nm:]*1e9, -(mag_90nm-meq_90nm)[t_TC_90nm:], lw=2.0, color='purple')
        axs[1][1].hlines(y=0, xmin=-0.1, xmax=5, color='black', ls='dashed', alpha=0.8)
        axs[1][1].yaxis.tick_right()
        # axs[1][1].set_ylabel(r'm$_{eq}$(t)-m(t)', fontsize=16)

        axs[2][0].plot(t_90nm[:t_TC_90nm]*1e9, te_90nm[:t_TC_90nm], lw=2.0, label=r'T$_e$', color='red')
        axs[2][0].plot(t_90nm[:t_TC_90nm]*1e9, tp_90nm[:t_TC_90nm], lw=2.0, label=r'T$_p$', color='darkgreen')
        axs[2][0].set_ylabel(r'Temperature [K]', fontsize=16)
        # axs[2][0].set_xlabel(r'delay [ns]', fontsize=16)
        axs[2][0].hlines(y=65, xmin=-0.1, xmax=5, lw=1.5, color='black', ls='solid', label=r'T$_C$')
        # axs[2][0].legend(fontsize=14, loc='center right')

        axs[2][1].plot(t_90nm[t_TC_90nm:]*1e9, te_90nm[t_TC_90nm:], lw=2.0, label=r'T$_e$', color='red')
        axs[2][1].plot(t_90nm[t_TC_90nm:]*1e9, tp_90nm[t_TC_90nm:], lw=2.0, label=r'T$_p$', color='darkgreen')
        # axs[2][1].set_ylabel(r'Temperature [K]', fontsize=16)
        axs[2][1].yaxis.tick_right()
        axs[2][1].set_xlabel(r'delay [ns]', fontsize=16)
        axs[2][1].hlines(y=65, xmin=-0.1, xmax=5, lw=1.5, color='black', ls='solid', label=r'T$_C$')
        axs[2][1].legend(fontsize=14, loc='center right')
        plt.savefig('Results/CGT Paper/90nm_review_reply.pdf')
        plt.show()

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
            data[:, 0] -= 0.2
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

        plt.figure(figsize=(10, 7))

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
            labels = ['15 nm CGT', '150 nm CGT']
            for loop_mat, loop_file, label in zip(mats, file, labels):
                exp_data = SimAnalysis.get_umd_data(loop_mat)
                sim_data = SimPlot(loop_file)
                delay, tes, tps, mags = sim_data.get_data()[:4]
                if loop_mat == 'cgt_thin':
                    mags = SimAnalysis.get_kerr(mags=mags, pen_dep=2e-9, layer_thick=14e-9, norm=True)
                else:
                    mags = SimAnalysis.get_kerr(mags=mags, pen_dep=2e-9, layer_thick=1e-9, norm=True)

                plt.scatter(exp_data[0], exp_data[1], s=4.0)
                plt.plot(delay * 1e12, mags, label=label, lw=2.0)
                plt.legend(fontsize=14)

        else:
            exp_data = SimAnalysis.get_umd_data(mat)
            sim_data = SimPlot(file).get_data()[:4]
            delay, tes, tps, mags = sim_data[:4]
            mags /= mags[0, 0]

            plt.scatter(exp_data[0], exp_data[1], marker='1', label=r'Lichtenberg et al.', s=130, color='green')
            plt.plot(delay*1e12, mags[:, 0]-1, lw=2.0, label=r'simulation', color='green')

        plt.xlim(-1, 5)
        plt.legend(fontsize=20)
        plt.xlabel(r'delay [ps]', fontsize=24)
        plt.ylabel(r'Kerr signal', fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig('DPG/FGT_fit.pdf')

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
    def fit_mag_tau1(file, t1, kind, save_fig, show_fig, save_folder=None):
        # This method fits the average of a simulation set of magnetization dynamics on long timescales after
        # e-p-equilibration to a function (1):
        # m(t) = const. + a*exp(-t/tau_1)+b*tanh((t-t_1)/tau_2)
        # or
        # m(t)=meq/sqrt(1-(1-meq**2)*exp(-2t/tau_2))

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

        # find time of minimum magetization and restrict times, magnetization data up to this point
        mmin_time_index = finder_nosort(np.amin(mag_av), mag_av)[0]
        delay_phase_1 = sim_delay[first_time_index:mmin_time_index]
        mag_phase_1 = mag_av[first_time_index:mmin_time_index]

        # if fitted with kind='LLB' function, the following part will activate:
        if kind == 'LLB':
            # initial guesses:
            tau_1_try = 5e-10
            p0_1 = [mag_phase_1[-1], tau_1_try, 2e-12]

            # the fit function:
            def LLB_demag(t, mmag, tau_1, t_offset):
                denom = np.sqrt(1-(1-mmag**2)*np.exp(-(t-t_offset)/tau_1))
                return mmag/denom

            lower_bounds = [mag_phase_1[-1]-0.05, 1e-15, 0]
            upper_bounds = [1, 1e-11, 5e-12]

            # run the fit:
            popt_1, cv_1 = scipy.optimize.curve_fit(LLB_demag, delay_phase_1, mag_phase_1, p0_1, bounds=(lower_bounds, upper_bounds))

            # if the fit does not converge, lower the guess of tau_1:
            while not np.all(np.isfinite(cv_1)):
                print('trying again: ', file)
                tau_1_try *= 0.9
                p0_1 = [mag_av[mmin_time_index + first_time_index], tau_1_try, 2e-12]
                popt_1, cv_1 = scipy.optimize.curve_fit(LLB_demag, delay_phase_1, mag_phase_1, p0_1, bounds=(lower_bounds, upper_bounds))

            # print the results
            print('simulation file: ', file)
            print('fit_mmag, fit_tau_1, fit_t_offset = ', popt_1)
            print('covariance = ', np.sqrt(np.diag(cv_1)))
            fit_string = 'LLB'

            plt.plot(delay_phase_1, mag_phase_1, label='sim', ls='dotted', color='orange')
            plt.plot(delay_phase_1, LLB_demag(delay_phase_1, *popt_1), label='fit')

            plt.legend(fontsize=14)
            plt.xlabel(r'delay [s]', fontsize=16)
            plt.ylabel(r'average magnetization', fontsize=16)

        # if fitted with kind='exp' function, the following will activate:
        if kind == 'exp':
            # initial guesses:
            tau_1_try = 5e-11
            p0_1 = [mag_phase_1[-1], tau_1_try, mag_phase_1[0]]
            lower_bounds = [0, 1e-13, 0]
            upper_bounds = [1, 1e-9, 1]

            # the fit function:
            def exp_decay(t, meq, tau_1, m0):
                return (m0-meq)*np.exp(-(t-delay_phase_1[0])/tau_1)+meq

            # run the fit:
            popt_1, cv_1 = scipy.optimize.curve_fit(exp_decay, delay_phase_1, mag_phase_1, p0_1, bounds=(lower_bounds, upper_bounds))

            # if the fit does not converge, lower the guess of tau_1:
            while not np.all(np.isfinite(cv_1)):
                print('trying again: ', file)
                tau_1_try *= 0.9
                p0_1 = [mag_av[mmin_time_index + first_time_index], tau_1_try, 2e-12]
                popt_1, cv_1 = scipy.optimize.curve_fit(exp_decay, delay_phase_1, mag_phase_1, p0_1, bounds=(lower_bounds, upper_bounds))

            # print the results:
            print('simulation file:', file)
            print('fit_mmag, fit_tau_1, fit_m0 = ', popt_1)
            print('covariance = ', np.sqrt(np.diag(cv_1)))
            fit_string = 'exp'

            plt.plot(delay_phase_1, mag_phase_1, label='sim', ls='dotted', color='orange')
            plt.plot(delay_phase_1, exp_decay(delay_phase_1, *popt_1), label='fit')

            plt.legend(fontsize=14)
            plt.xlabel(r'delay [s]', fontsize=16)
            plt.ylabel(r'average magnetization', fontsize=16)

        if show_fig:
            plt.show()

        if save_fig:
            assert save_folder is not None, 'Please give a foldername to save the plots with save_folder=...'
            plt.savefig(save_folder + '/' + file + '_decay_' + fit_string + '.pdf')

        return popt_1, cv_1

    @staticmethod
    def fit_mag_tau2(file, kind, save_fig, show_fig, save_folder=None, end_time=None):

        # get the data to fit:
        sim_data = SimPlot(file).get_data()
        sim_delay = sim_data[0]
        sim_mags = sim_data[3]

        # average the magnetization:
        mag_av = np.sum(sim_mags, axis=1) / len(sim_mags[0])

        # find time of minimum magnetization and restrict times, magnetization data to this point and later
        mmin_time_index = finder_nosort(np.amin(mag_av), mag_av)[0]
        delay_phase_2 = sim_delay[mmin_time_index:]
        mag_phase_2 = mag_av[mmin_time_index:]

        if end_time is None:
            end_time_index = len(delay_phase_2) + 1
        else:
            end_time_index = finderb(end_time, delay_phase_2)[0]

        # find the time of the flex point during remagnetization (d^2m/dt^2 = 0)
        d2m = np.diff(mag_phase_2, 2)
        flex_index = -1

        for i, el in enumerate(d2m):
            if np.abs(el + d2m[i+1]) <= np.abs(el-d2m[i+1]):
                flex_index = i
                print('start_time of remagnetization fit (flex point): ', delay_phase_2[i]*1e9, ' ns')
                break

        assert flex_index != -1, 'The flex point of the remagnetization could not be determined. Aborting the fit.'

        # restrict the fit to flex_point:end_time
        delay_phase_2 = delay_phase_2[flex_index:end_time_index]
        mag_phase_2 = mag_phase_2[flex_index:end_time_index]

        if kind == 'exp':
            # define the fit function
            def exp_remag(t, A, tau_2, t_offset, meq):
                return A * np.exp(-(t-t_offset)/tau_2) + meq

            # initial guesses:
            tau_2_try = 1e-9
            p0 = [(mag_phase_2[0]-mag_phase_2[-1]), tau_2_try, delay_phase_2[0], mag_phase_2[-1]]

            # run the fit:
            popt, cv = scipy.optimize.curve_fit(exp_remag, delay_phase_2, mag_phase_2, p0)

            # if the fit does not converge, lower the guess of tau_1:
            while not np.all(np.isfinite(cv)):
                print('trying again: ', file)
                tau_2_try *= 0.9
                p0 = [(mag_phase_2[0]-mag_phase_2[-1]), tau_2_try, delay_phase_2[0], mag_phase_2[-1]]
                popt, cv = scipy.optimize.curve_fit(exp_remag, delay_phase_2, mag_phase_2, p0)


            print('offset, tau_2, t_offset, meq = ', popt)
            print('covariance = ', np.sqrt(np.diag(cv)))

            plt.plot(delay_phase_2, mag_phase_2, ls='dotted', color='orange')
            plt.plot(delay_phase_2, exp_remag(delay_phase_2, *popt), label='fit')
            # plt.plot(delay_phase_2, exp_remag(delay_phase_2, *p0), label='default')

            plt.legend(fontsize=14)
            plt.xlabel(r'delay [s]', fontsize=16)
            plt.ylabel(r'average magnetization', fontsize=16)
            fit_string = 'exp'

        # if fitted with kind='LLB' function, the following part will activate:
        if kind == 'LLB':
            # initial guesses:
            tau_2_try = 1e-10
            p0 = [0.9, tau_2_try, delay_phase_2[0]]

            # the fit function:
            def LLB_remag(t, mmag, tau_2, t_offset):
                denom = -np.sqrt(1 - (1 - mmag ** 2) * np.exp(-2 * (t - delay_phase_2[0]) / tau_2))
                return mmag / denom + 1+ mag_phase_2[0]

            # # run the fit:
            # popt, cv = scipy.optimize.curve_fit(LLB_remag, delay_phase_2, mag_phase_2, p0)
            #
            # # if the fit does not converge, lower the guess of tau_1:
            # while not np.all(np.isfinite(cv)):
            #     print('trying again: ', file)
            #     tau_2_try *= 0.9
            #     p0 = [mag_phase_2[-1], tau_2_try, delay_phase_2[0]]
            #     popt, cv = scipy.optimize.curve_fit(LLB_remag, delay_phase_2, mag_phase_2, p0)

            # print(file)
            # print('mmag, tau_2, t_offset = ', popt)
            # print('covariance = ', np.sqrt(np.diag(cv)))

            plt.plot(delay_phase_2, mag_phase_2, ls='dotted', color='orange')
            # plt.plot(delay_phase_2, LLB_remag(delay_phase_2, *popt), label='fit')
            plt.plot(delay_phase_2, LLB_remag(delay_phase_2, *p0), label='default')

            plt.legend(fontsize=14)
            plt.xlabel(r'delay [s]', fontsize=16)
            plt.ylabel(r'average magnetization', fontsize=16)
            fit_string = 'LLB'

        if show_fig:
            plt.show()

        if save_fig:
            assert save_folder is not None, 'Please give a foldername to save the plots with save_folder=...'
            plt.savefig(save_folder + '/' + file + '_remag_' + fit_string + '.pdf')

        return popt, cv

    @staticmethod
    def fit_mag_data(file, t1, t2):
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
        # sim_data = SimPlot(file).get_data()
        # sim_delay = sim_data[0]
        # sim_mags = sim_data[3]

        data = np.loadtxt('C:/Users/Theodor Griepe/Documents/Github/Magneto-Thermal-Engineering/sim_to_fit/FGT_AIN.dat')
        sim_delay = (data[:, 0]-1)*1e-12
        mag_av = data[:, 1]
        mag_av = (mag_av[0]-mag_av)/mag_av[0]
        # average the magnetization:
        # mag_av = np.sum(sim_mags, axis=1) / len(sim_mags[0])

        # sort out the time intervals
        first_time_index = finderb(t1, sim_delay)[0]
        last_time_index = finderb(t2, sim_delay)[0]

        delay_phase_12 = sim_delay[first_time_index:last_time_index]
        mag_phase_12 = mag_av[first_time_index:last_time_index]

        mag_diff = np.diff(mag_av)
        time_diff = np.diff(sim_delay)
        dm_dt = mag_diff / time_diff

        def exp3(t, tau_e, dtau_eA, dA_12, A_1, tau_3, A_4, tau_4):
            # define the missing parameters with respect to the given ones:
            A_2 = A_1 + dA_12
            tau_m = tau_e*(1+dA_12/A_1)+dtau_eA

            # define the terms 2,3,4 and add them to get an array of length of len(t):
            term_ep = -tau_e*(A_1-A_2)/(tau_e-tau_m)*np.exp(-t/tau_e)
            term_mag = -(A_2*tau_e-A_1*tau_m)/(tau_e-tau_m)*np.exp(-t/tau_m)
            term_remag_1 = A_1*np.exp(-t/tau_3)
            term_remag_2 = A_4*np.exp(-t/tau_4)
            combined_func = term_ep + term_mag + term_remag_1 + term_remag_2
            return combined_func

        def exp4(t, tau_e, dtau_me, tau_re1, tau_re2, A_e, A_m, A_re1, A_re2):
            tau_m = tau_e + dtau_me
            term_ep = A_e*np.exp(-t/tau_e)
            term_em = A_m*np.exp(-t/tau_m)
            term_re1 = A_re1*np.exp(-t/tau_re1)
            term_re2 = A_re2*np.exp(-t/tau_re2)

            combined = term_ep + term_em + term_re1 + term_re2
            return combined

        lower_bounds4 = [0., 0., 0., 0., -np.inf, -np.inf, 0., 0.]
        upper_bounds4 = [1e-11, 8e-10, 1e-9, 1e-8, 0, 0, np.inf, np.inf]
        bounds4 = np.array([lower_bounds4, upper_bounds4])
        p0_all4 = [2e-13, 6e-12, 1e-10, 1e-9, -3e-2, -8e-3, 2e-2, 1e-2]
        popt_all4, cv_all4 = scipy.optimize.curve_fit(exp4, delay_phase_12, mag_phase_12, p0_all4,
                                                    bounds=bounds4)
        plt.plot(delay_phase_12, exp4(delay_phase_12, *popt_all4), label='fit')
        plt.plot(delay_phase_12, exp4(delay_phase_12, *p0_all4), label='initial')

        # lower_bounds3 = [0, 0, 0, 0, 0, 0, 0]
        # upper_bounds3 = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        # bounds3 = np.array([lower_bounds3, upper_bounds3])
        # p0_all3 = [0.1e-12, 3e-12, 1e-2, 0.03, 0.8e-10, 1e-3, 1e-9]
        # popt_all3, cv_all3 = scipy.optimize.curve_fit(exp3, delay_phase_12, mag_phase_12, p0_all3,
        #                                             bounds=bounds3)
        # plt.plot(delay_phase_12, exp3(delay_phase_12, *popt_all3), label='fit')
        # plt.plot(delay_phase_12, exp3(delay_phase_12, *p0_all3), label='initial')

        plt.plot(delay_phase_12, mag_phase_12, label='sim')
        plt.legend()
        plt.show()

        # print('a1,a3, tau0, tau2, tau3, e_tau, e_tau_a = ', popt_all3)

        # # Residual analysis
        # residuals = mag_phase_12 - exp3(delay_phase_12, *popt_all)
        # plt.plot(delay_phase_12, residuals, label='Residuals', color='red')
        # plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        # plt.xlabel('Delay [s]')
        # plt.ylabel('Residuals')
        # plt.legend()
        # plt.show()
        # # Goodness-of-fit metrics calculation
        # y_pred = exp3(delay_phase_12, *popt_all)
        # SS_res = np.sum((mag_phase_12 - y_pred) ** 2)
        # SS_tot = np.sum((mag_phase_12 - np.mean(mag_phase_12)) ** 2)
        # R_squared = 1 - (SS_res / SS_tot)
        # RMSE = np.sqrt(np.mean((mag_phase_12 - y_pred) ** 2))
        # print('R^2:', R_squared)
        # print('RMSE:', RMSE)
        #
        # plt.figure(figsize=(10, 8))
        # ax = plt.gca()
        # plt.plot(sim_delay * 1e9, mag_av, color='black', marker='o', linestyle='', markersize=12, markevery=200,
        #          markerfacecolor='none')
        # plt.plot(delay_phase_12 * 1e9, exp3(delay_phase_12, *popt_all), color='royalblue', ls='-', linewidth='6')
        #
        # plt.xlabel(r'$\mathrm{Delay (ns)}$', fontsize=40)
        # plt.ylabel(r'$\mathrm{M/M_0}$', fontsize=40)
        # plt.tick_params(which='major', length=8)
        # plt.text(3.5, 0.75, '$\mathdefault{Graphene}$', fontsize=40, color='red')
        # plt.tight_layout()
        # plt.show()

        return popt_all, np.sqrt(np.diag(cv_all))

    def fit_all_mag_tau1(self, save_file_name, t1, kind, save_fig, show_fig, save_folder=None, end_time=None):
        # create an array to store all fit parameters: ferro X sub X fluence X param
        all_mag_params = np.zeros((3, 5, 2, 3), dtype=float)
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

            if '14' in file:  # you might put thickness here
                k = 0
            elif '90' in file:  # you might put thickness here
                k = 1
            else:
                print('Did not find the thickness you made me search')

            popt, cv = SimAnalysis.fit_mag_tau1(file=file, t1=t1, show_fig=True, save_fig=True, save_folder=save_folder, kind=kind)
            # popt = T0, Teq, exponent, delay
            all_mag_params[i, j, k] = popt

        np.save('Results/' + save_file_name + '.npy', all_mag_params)
        return

    def fit_all_mag_tau2(self, save_file_name, kind, save_fig, show_fig, save_folder=None, end_time=None):
        # create an array to store all fit parameters: ferro X sub X fluence X param
        all_mag_params = np.zeros((3, 5, 2, 4), dtype=float)
        # determine where to save:
        for file in self.files:

            if 'FGT' in file:
                i = 0
            elif 'CGT' in file:
                i = 1
            elif 'CrI3' in file:
                i = 2
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

            if '14' in file:  # you might put thickness here
                k = 0
            elif '90' in file:  # you might put thickness here
                k = 1
            else:
                print('Did not find the thickness you made me search')

            popt, cv = SimAnalysis.fit_mag_tau2(file=file, kind=kind, save_fig=save_fig, show_fig=show_fig, save_folder=save_folder, end_time=end_time)
            # popt = T0, Teq, exponent, delay
            all_mag_params[i, j, k] = popt

        np.save('Results/' + save_file_name + '.npy', all_mag_params)
        return

    @staticmethod
    def plot_tau_1_params(file, savefile):

        # load the array:
        all_phonon_params = np.load('Results/' + file)

        # constant fluence, all substrates, each material:
        mmag_params = [[[], []], [[], []], [[], []]]
        tau_1_params = [[[], []], [[], []], [[], []]]
        t_offset_params = [[[], []], [[], []], [[], []]]

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
                for k, param in enumerate(sub_row[0, :]):  # the 1 here refers to either thin (m) or thick (n) sample
                    if k == 2:
                        mmag_params[i][0].append(kappa)
                        mmag_params[i][1].append(param)
                    elif k == 0:
                        tau_1_params[i][0].append(kappa)
                        tau_1_params[i][1].append(param*1e9)
                    elif k == 1:
                        t_offset_params[i][0].append(kappa)
                        t_offset_params[i][1].append(param*1e9)

        teq_params = np.array(mmag_params)
        tau_1_params = np.array(tau_1_params)
        t_offset_params = np.array(t_offset_params)
        # create figure:
        fig = plt.figure(layout='constrained', figsize=(7, 9))
        fig_mmag = plt.subplot(3, 1, 1)
        fig_tau_1 = plt.subplot(3, 1, 2)
        fig_t_offset = plt.subplot(3, 1, 3)

        # arange figure:
        fig_t_offset.set_xlabel(r'substrate $\kappa$ [W/mK]', fontsize=20)

        fig_mmag.set_ylabel(r'$m_{eq}}$', fontsize=20)
        fig_tau_1.set_ylabel(r'$\tau_{1}$ [ns]', fontsize=20)
        fig_t_offset.set_ylabel(r't_offset [ns]', fontsize=20)

        fig_t_offset.tick_params(axis='x', labelsize=18)
        fig_mmag.xaxis.set_tick_params(labelbottom=False)
        fig_tau_1.xaxis.set_tick_params(labelbottom=False)

        fig_mmag.tick_params(axis='y', labelsize=18)
        fig_tau_1.tick_params(axis='y', labelsize=18)
        fig_t_offset.tick_params(axis='y', labelsize=18)

        colors = [np.array([185, 132, 140])/255, np.array([47, 112, 175])/255, np.array([128, 100, 145])/255]
        labels = [r'FGT', r'CGT', r'CrI3']

        for i in range(len(colors)):
            fig_mmag.plot(mmag_params[i][0], mmag_params[i][1], ls='dashed', marker='o', lw=0.5, color=colors[i], label=labels[i])
            fig_tau_1.plot(tau_1_params[i][0], (tau_1_params[i][1]), ls='dashed', marker='o', lw=0.5, color=colors[i])
            fig_t_offset.plot(t_offset_params[i][0], t_offset_params[i][1], ls='dashed', marker='o', lw=0.5, color=colors[i])

        fig_mmag.legend(fontsize=18)
        plt.savefig('Results/' + savefile + '.pdf')

        plt.show()
        return

    @staticmethod
    def plot_tau_2_params(file, savefile):

        # load the array:
        all_phonon_params = np.load('Results/' + file)

        # constant fluence, all substrates, each material:
        amp_params = [[[], []], [[], []], [[], []]]
        tau_2_params = [[[], []], [[], []], [[], []]]
        t_offset_params = [[[], []], [[], []], [[], []]]
        mmag_params = [[[], []], [[], []], [[], []]]

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
                for k, param in enumerate(sub_row[1, 2:]):  # the 1 here refers to either thin (m) or thick (n) sample
                    if k == 2:
                        amp_params[i][0].append(kappa)
                        amp_params[i][1].append(param)
                    elif k == 0:
                        tau_2_params[i][0].append(kappa)
                        tau_2_params[i][1].append(param * 1e9)
                    elif k == 1:
                        t_offset_params[i][0].append(kappa)
                        t_offset_params[i][1].append(param * 1e9)
                    elif k == 2:
                        mmag_params[i][0].append(kappa)
                        mmag_params[i][1].append(param)

        amp_params = np.array(amp_params)
        tau_2_params = np.array(tau_2_params)
        t_offset_params = np.array(t_offset_params)
        mmag_params = np.array(mmag_params)

        # create figure:
        fig = plt.figure(layout='constrained', figsize=(7, 9))
        fig_amp = plt.subplot(4, 1, 1)
        fig_tau_2 = plt.subplot(4, 1, 2)
        fig_t_offset = plt.subplot(4, 1, 3)
        fig_mmag = plt.subplot(4, 1, 4)

        # arange figure:
        fig_mmag.set_xlabel(r'substrate $\kappa$ [W/mK]', fontsize=20)

        fig_mmag.set_ylabel(r'$m_{eq}}$', fontsize=20)
        fig_tau_2.set_ylabel(r'$\tau_{2}$ [ns]', fontsize=20)
        fig_t_offset.set_ylabel(r't_offset [ns]', fontsize=20)
        fig_amp.set_ylabel(r'amplitude', fontsize=20)

        fig_mmag.tick_params(axis='x', labelsize=18)
        fig_t_offset.xaxis.set_tick_params(labelbottom=False)
        fig_tau_2.xaxis.set_tick_params(labelbottom=False)
        fig_amp.xaxis.set_tick_params(labelbottom=False)

        fig_mmag.tick_params(axis='y', labelsize=18)
        fig_tau_2.tick_params(axis='y', labelsize=18)
        fig_t_offset.tick_params(axis='y', labelsize=18)
        fig_amp.tick_params(axis='y', labelsize=18)

        colors = [np.array([185, 132, 140]) / 255, np.array([47, 112, 175]) / 255, np.array([128, 100, 145]) / 255]
        labels = [r'FGT', r'CGT', r'CrI3']

        for i in range(len(colors)):
            fig_mmag.plot(mmag_params[i][0], mmag_params[i][1], ls='dashed', marker='o', lw=0.5, color=colors[i],
                          label=labels[i])
            fig_tau_2.plot(tau_2_params[i][0], (tau_2_params[i][1]), ls='dashed', marker='o', lw=0.5, color=colors[i])
            fig_t_offset.plot(t_offset_params[i][0], t_offset_params[i][1], ls='dashed', marker='o', lw=0.5,
                              color=colors[i])
            fig_amp.plot(amp_params[i][0], amp_params[i][1], ls='dashed', marker='o', lw=0.5,
                              color=colors[i])

        fig_mmag.legend(fontsize=18)
        plt.savefig('Results/' + savefile + '.pdf')

        plt.show()
        return
