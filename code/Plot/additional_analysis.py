import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import constants as sp
from scipy import interpolate as ip
from scipy import optimize as op
from matplotlib import colors as mplcol

from .plot import SimComparePlot
from ..Source.finderb import finderb

class SimAnalysis(SimComparePlot):
    def __init__(self, files):
        super().__init__(files)

    def plot_spin_acc(self, labels, save_path):
        # This method plots the magnetization rate dm/dt of a list of simulations over time, normalized to the maximum
        # rate of the FIRST simulation given, so self.files[0]. The plot is divided into four subplots to allow for
        # different zoom effects in time and amplitude.

        # Input:
        # self (class object). The object in use. Specifically, self.files must be defined
        # labels (list of strings). Labels for the simulation data
        # save_path. A relative path from 'Results/' denoting where the plot should be saved

        # Returns:
        # None. void function

        # create a 4x4 figure
        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', width_ratios=[1, 5], figsize=(8, 5))

        # define axes labels and ticks
        axs[1][1].set_xlabel(r'delay [ns]', fontsize=16)
        axs[1][1].set_ylabel(r'norm. dm/dt', fontsize=16)
        axs[1][1].yaxis.set_label_position("right")
        axs[0][1].hlines(0, -0.1, 5, color='black', lw=1.5, ls='dashed')
        axs[0][0].hlines(0, -0.1, 5, color='black', lw=1.5, ls='dashed')

        # no lines between subplots top and bottom
        axs[0][0].spines.bottom.set_visible(False)
        axs[0][1].spines.bottom.set_visible(False)
        axs[1][1].spines.top.set_visible(False)
        axs[1][0].spines.top.set_visible(False)

        # adjust ticks
        axs[0][0].xaxis.tick_top()
        axs[0][1].xaxis.tick_top()
        axs[0][0].tick_params(labeltop=False)
        axs[1][0].tick_params(labeltop=False)
        axs[1][0].xaxis.tick_bottom()
        axs[1][1].xaxis.tick_bottom()
        axs[0][1].yaxis.tick_right()
        axs[1][1].yaxis.tick_right()

        # restrict data shown
        axs[0][0].set_xlim(-1e-3, 1e-2)  # top left ([0][0]) and bottom left ([1][0]) x limits
        axs[0][0].set_ylim(0, 0.01)  # top left ([0][0]) and top right ([0][1]) y limits
        axs[1][0].set_ylim(-1.05, -1e-5)  # bottom left ([1][0]) and bottom right ([1][1]) y limits
        axs[1][1].set_xlim(0.01, 5)  # bottom right ([1][1]) and top right ([0][1]) y limits

        # no gaps between subplots
        plt.subplots_adjust(wspace=0, hspace=0)

        # check amount of labels:
        assert len(labels) == len(self.files), 'Introduce as many labels as datasets you want to plot'
        norm = 1

        # plot all data:
        for i, file in enumerate(self.files):
            delays, mags, tes, tps = self.get_data(file)

            mag_av = np.sum(mags, axis=1) / len(mags[0])
            dmag_dt = np.diff(mag_av)/np.diff(delays)
            dtt = np.cumsum(np.diff(delays*1e9))
            te_av = np.sum(tes, axis=1) / len(tes[0])

            # THE FIRST FILE IN SELF.FILES SHOULD BE THE ONE WITH THE HIGHEST DEMAGNETIZATION RATE
            # TO GET PROPER NORMALIZATION
            if i == 0:
                norm = np.amax(np.abs(dmag_dt))
            # PLOT THE DATA IN EACH QUADRANT. THE LIMITS OF THE QUADRANTS AXES (ABOVE) DETERMINE WHAT IS SHOWN
            axs[0][0].plot(dtt-0.001, dmag_dt/norm, lw=2.0, label=labels[i])
            axs[0][1].plot(dtt-0.001, dmag_dt/norm, lw=2.0, label=labels[i])
            axs[1][0].plot(dtt-0.001, dmag_dt/norm, lw=2.0, label=labels[i])
            axs[1][1].plot(dtt-0.001, dmag_dt/norm, lw=2.0, label=labels[i])

        axs[1][1].legend(fontsize=14)
        plt.savefig('Results/' + save_path)
        plt.show()

        return

    @staticmethod
    def Brillouin(temps, mags, spin, Tc):
        # This function computes the Brillouin function.

        # Input:
        # temps (array/list/float). (electron) temperature
        # mags (array/list/float). magnetization in same dimension as temps
        # spin (float). effective spin of material
        # Tc (float). Curie temperature of material in K.

        # Returns:
        # term_1 + term_2 (array/list/float). brillouin function at the given input values
        pref_1 = (2 * spin + 1) / (2 * spin)
        pref_2 = 1 / (2 * spin)
        x = 3 * Tc * mags / temps * spin / (spin + 1)

        term_1 = pref_1 / np.tanh(pref_1 * x)
        term_2 = -pref_2 / np.tanh(pref_2 * x)

        return term_1 + term_2

    @staticmethod
    def create_mean_mag_map(S, Tc):
        # This function computes the mean field mean magnetization map by solving the self-consistent equation m=B(m, T)
        # As an output we get an interpolation function of the mean field magnetization at any temperature T<= 100*T_c.

        # Input:
        # S (float). effective spin of material
        # Tc (float). Curie temperature of material

        # Returns:
        # ip.interp1d(temp_grid, meq_list) (interpolation object). function that takes temperature inputs
        # normalized to T_c and returns the corresponding mean field equilibrium magnetization.

        # Start by defining a unity function m=m:
        def mag(m):
            return m

        # Define the Brillouin function as a function of scalars, as fsolve takes functions of scalars:
        def Brillouin(m, T):
            # This function takes input parameters
            #   (i) magnetization amplitude m_amp_grid (scalar)
            #   (ii) (electron) temperature (scalar)
            # As an output we get the Brillouin function evaluated at (i), (ii) (scalar)

            J = 3 * S / (S + 1) * sp.k * Tc
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
            if meq[
                0] < 0:  # This is a comletely unwarranted fix for values of meq<0 at temperatures very close to Tc, that fsolve produces. It seems to work though, as the interpolated function plotted by plot_mean_mags() seems clean.
                meq[0] *= -1
            # Append it to list me(T)
            meq_list.append(meq[0])
        meq_list[
            -1] = 0  # This fixes slight computational errors to fix m_eq(Tc)=0 (it produces something like m_eq[-1]=1e-7)
        temp_grid = np.append(temp_grid, 100.)
        meq_list.append(0.)
        return ip.interp1d(temp_grid, meq_list)

    @staticmethod
    def plot_m_meq(file, num_cgt_layers, save_file_name):
        # This method creates a plot like figure 1 in the document for review response.

        # Input:
        # file (string). Filepath relative to Results of the simulation data you want to plot
        # num_cgt_layers (int). Number of CGT layers in the simulation
        # save_file_sanme (string). Filepath to save the figure relative to Results.

        # create mean mag map:
        meq = SimAnalysis.create_mean_mag_map(S=1.5, Tc=65.)
        # load data:
        t = np.load('Results/' + file + '/delay.npy')
        te_raw = np.load('Results/' + file + '/tes.npy')
        mag_raw = np.load('Results/' + file + '/ms.npy')
        tp_raw = np.load('Results/' + file + '/tps.npy')
        te = np.sum(te_raw, axis=1) / num_cgt_layers
        mag = np.sum(mag_raw, axis=1) / num_cgt_layers
        tp = np.sum(tp_raw[:, 7:7 + num_cgt_layers], axis=1) / num_cgt_layers

        # find meq(Te(t)):
        meq = meq(te / 65.)
        bs = SimAnalysis.Brillouin(temps=te, mags=mag, spin=1.5, Tc=65.)

        time_to_break = 0.01
        t_TC = finderb(time_to_break, t * 1e9)[0]
        fig, axs = plt.subplots(3, 2, sharex='col', figsize=(8, 6), width_ratios=[1, 5], layout='compressed')

        axs[0][0].plot(t[:t_TC] * 1e12, mag[:t_TC], lw=2.0)
        axs[0][0].plot(t[:t_TC] * 1e12, bs[:t_TC], lw=2.0)
        axs[0][0].plot(t[:t_TC] * 1e12, meq[:t_TC], lw=2.0, alpha=0.4, color='black')
        axs[0][0].set_ylabel(r'magnetization', fontsize=16)
        axs[0][0].set_xlim(-0.1, time_to_break * 1e3)
        axs[0][0].set_ylim(-0.01, 1.01)

        axs[0][1].plot(t[t_TC:] * 1e9, mag[t_TC:], lw=2.0, label=r'm(t)')
        axs[0][1].plot(t[t_TC:] * 1e9, bs[t_TC:], lw=2.0, label=r'B$_S$(m(t), Te(t))')
        axs[0][1].plot(t[t_TC:] * 1e9, meq[t_TC:], lw=2.0, alpha=0.4, color='black', label=r'm$_{\rm{eq}}(t)$')
        axs[0][1].yaxis.tick_right()
        axs[0][1].set_xlim(time_to_break, 5)
        axs[0][1].legend(fontsize=14, loc='upper center')
        axs[0][1].set_ylim(-0.01, 1.01)

        axs[1][0].plot(t[:t_TC] * 1e12, (bs - mag)[:t_TC], lw=2.0, color='purple')
        axs[1][0].hlines(y=0, xmin=-0.01, xmax=time_to_break * 1e3, color='black', ls='dashed', alpha=0.8)
        axs[1][0].set_ylabel(r'B$_S$(t)-m(t)', fontsize=16)

        axs[1][1].plot(t[t_TC:] * 1e9, (bs - mag)[t_TC:], lw=2.0, color='purple')
        axs[1][1].hlines(y=0, xmin=-0.1, xmax=5, color='black', ls='dashed', alpha=0.8)
        axs[1][1].yaxis.tick_right()

        axs[2][0].plot(t[:t_TC] * 1e12, te[:t_TC], lw=2.0, label=r'T$_e$', color='red')
        axs[2][0].plot(t[:t_TC] * 1e12, tp[:t_TC], lw=2.0, label=r'T$_p$', color='darkgreen')
        axs[2][0].set_ylabel(r'Temperature [K]', fontsize=16)
        axs[2][0].set_xlabel(r'delay [ps]', fontsize=16)
        axs[2][0].hlines(y=65, xmin=-0.01, xmax=time_to_break * 1e3, lw=1.5, color='black', ls='solid', label=r'T$_C$')

        axs[2][1].plot(t[t_TC:] * 1e9, te[t_TC:], lw=2.0, label=r'T$_e$', color='red')
        axs[2][1].plot(t[t_TC:] * 1e9, tp[t_TC:], lw=2.0, label=r'T$_p$', color='darkgreen')
        axs[2][1].yaxis.tick_right()
        axs[2][1].set_xlabel(r'delay [ns]', fontsize=16)
        axs[2][1].hlines(y=65, xmin=-0.1, xmax=5, lw=1.5, color='black', ls='solid', label=r'T$_C$')
        axs[2][1].legend(fontsize=14, loc='center right')
        plt.savefig('Results/' + save_file_name)
        plt.show()

        return