import numpy as np
import os
from matplotlib import pyplot as plt


def get_data(self, subsys):

    sim_files_folder = 'Results/FGT/' + self.file + '/'
    delay = np.load(sim_files_folder + 'delays.npy')
    te = np.load(sim_files_folder + 'tes.npy')
    tp = np.load(sim_files_folder + 'tps.npy')
    mag = np.load(sim_files_folder + 'ms.npy')

    if os.path.isfile(sim_files_folder + 'tp2s.npy'):
        tp2 = np.load(sim_files_folder + 'tp2s.npy')
    else:
        tp2 = np.zeros_like(tp)

    return delay, te, tp, tp2, mag


def create_figure():

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].set_x_label(r'delay [ps]', fontsize=16)
    axs[0].set_ylabel(r'$m/m_0$', fontsize=16)
    axs[1].set_ylabel(r'$T_p$', fontsize=16)
    axs[2].set_ylabel(r'$T_e$', fontsize=16)

    axs[0].set_xlim(0, 10)
    axs[1].set_xlim(0, 10)
    axs[2].set_xlim(0, 10)

    fig.tight_layout()

    return fig, axs

def plot_te()

