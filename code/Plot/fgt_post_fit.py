import numpy as np
import os
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


def read_fit_file() -> dict:
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


def find_neighbouring_params(fit_dict: dict, param:str, smaller_or_bigger: str) -> dict:
    # find the "neighbouring" parameters to the optimal fit from the simulation files

    gammas = np.arange(205, 226).astype(float)  # 21 values
    t0_el = np.arange(20)  # 20 values
    geps = np.arange(4.0, 5.6, 0.1)  # 16 values
    asfs = np.arange(0.01, 0.021, 0.001)  # 11 values
    gpps = np.arange(2.5, 4.1, 0.1)  # 16 values
    k_ep = np.arange(5, 25) * 2e-2  # 20 values

    if smaller_or_bigger == "bigger":
        index_shift = 1
    elif smaller_or_bigger == "smaller":
        index_shift = -1
    else:
        print(f"{smaller_or_bigger} is not a proper input for smaller_or_bigger")
        exit()

    shifted_dict = fit_dict.copy()

    if param == "gamma":
        shifted_index = list(gammas).index(fit_dict["gamma"]) + index_shift
        shifted_dict["gamma"] = gammas[shifted_index]
    elif param == "asf":
        shifted_index = list(asfs).index(fit_dict["asf"]) + index_shift
        shifted_dict["asf"] = asfs[shifted_index]
    elif param == "gep":
        shifted_index = list(geps).index(fit_dict["gep"]) + index_shift
        shifted_dict["gep"] = geps[shifted_index]
    elif param == "gpp":
        shifted_index = list(gpps).index(fit_dict["gpp"]) + index_shift
        shifted_dict["gpp"] = gpps[shifted_index]
    elif param == "t0":
        shifted_index = list(t0_el).index(fit_dict["t0"]) + index_shift
        shifted_dict["t0"] = t0_el[shifted_index]
    elif param == "k":
        shifted_index = list(k_ep).index(fit_dict["k"]) + index_shift
        shifted_dict["k"] = k_ep[shifted_index]
    else:
        print(f"{param} is not a valid input for param")

    return shifted_dict


def get_filename_from_params(param_dict: dict) -> str:
    # find simulation file name corresponding to parameters from dictionary

    file_name = f"a{param_dict["asf"]}gep{param_dict["gep"]}gpp{param_dict["gpp"]}gamma{param_dict["gamma"]}"

    return file_name


def get_standard_deviations():

    opt_fit_params = read_fit_file()
    opt_fit_file = get_filename_from_params(opt_fit_params)

