import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mplcol
from scipy import constants as sp
from scipy import interpolate as ip

from .plot import SimComparePlot
from .plot import SimPlot


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
        else:
            exp_data = SimAnalysis.get_umd_data(mat)
            sim_data = SimPlot(file)
            delay, tes, tps, mags = sim_data.get_data()[:4]
            mags /= mags[0, 0]

            plt.figure(figsize=(8, 6))
            plt.scatter(exp_data[0], exp_data[1])
            plt.plot(delay*1e12, mags[:, 0]-1)

        # plt.xlim(-1, 25)

        plt.xlabel(r'delay [ps]', fontsize=16)
        plt.ylabel(r'magnetization', fontsize=16)

        plt.show()
        return
