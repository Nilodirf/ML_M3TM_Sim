import numpy as np
import os
from scipy import io
from scipy.stats import chi2
import time
import multiprocessing as mp

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
        val_1 = val_1 / val_1[0]
        val_2 = None
    else:
        print('Idiot, put te ot tp or mag for subsys')
        return

    return delay * 1e12 - 1, val_1, val_2


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

def read_fit_file():
    # reads the file with the best fit parameters

    fit_dict = {}

    with open("fit_values.dat", 'r') as file:
        content = file.readlines()

        for line in content:
            if line.startswith("gamma:"):
                fit_dict["gamma"] = float(line[line.find("gamma:" + 6):])
            if line.startswith("gep:"):
                fit_dict["gep"] = float(line[line.find("gep:" + 4):])
            if line.startswith("asf:"):
                fit_dict["asf"] = float(line[line.find("asf:" + 4):])
            if line.startswith("gpp:"):
                fit_dict["gpp"] = float(line[line.find("gpp:" + 4):])
            if line.startswith("t0:"):
                fit_dict["t0"] = float(line[line.find("t0:" + 3):])
            if line.startswith("k:"):
                fit_dict["k"] = float(line[line.find("k:" + 2):])

    return fit_dict

def find_neighbouring_files():
     
