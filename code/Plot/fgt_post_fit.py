import numpy as np
import os
from numpy.typing import NDArray
from scipy import io


def finderb(key, array):

    key = np.array(key, ndmin=1)
    n = len(key)
    i = np.zeros([n], dtype=int)

    for m in range(n):
        i[m] = finderb_nest(key[m], array)
    return i

def finderb_nest(key, array):

    a = 0  # start of intervall
    b = len(array)  # end of intervall

    # if the key is smaller than the first element of the
    # vector we return 1
    if key < array[0]:
        return 0

    while (b-a) > 1:  # loop until the intervall is larger than 1
        c = int(np.floor((a+b)/2))  # center of intervall
        if key < array[c]:
            # the key is in the left half-intervall
            b = c
        else:
            # the key is in the right half-intervall
            a = c

    return a

def get_data(file, subsys):
    sim_files_folder = 'Results/FGT/' + file + '/'
    delay = np.load(sim_files_folder + 'delay.npy')
    if subsys == "el":
        val_1 = np.load(sim_files_folder + 'tes.npy')[:, 0]
        val_2 = None
    elif subsys == "tp":
        val_1 = np.load(sim_files_folder + 'tps.npy')[:, 0]
        if os.path.isfile(sim_files_folder + 'tp2s.npy'):
            val_2 = np.load(sim_files_folder + 'tp2s.npy')[:, 0]
        else:
            val_2 = None
    elif subsys == "mag":
        val_1 = np.load(sim_files_folder + 'ms.npy')[:, 0]
        val_1 = val_1 / val_1[0]
        val_2 = None
    else:
        print('Input te ot tp or mag for subsys')
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

    with open("fit_values_t0_fs.dat", 'r') as file:
        content = file.readlines()

        for line in content:
            if line.startswith("gamma:"):
                fit_dict["gamma"] = float(line[line.find("gamma: ") + 7:])
            if line.startswith("gep:"):
                fit_dict["gep"] = float(line[line.find("gep: ") + 5:])
            if line.startswith("asf:"):
                fit_dict["asf"] = float(line[line.find("asf: ") + 5:])
            if line.startswith("gpp:"):
                fit_dict["gpp"] = float(line[line.find("gpp: ") + 5:])
            if line.startswith("t0:"):
                fit_dict["t0"] = float(line[line.find("t0: ") + 4:])
            if line.startswith("k:"):
                fit_dict["k"] = float(line[line.find("k: ") + 3:])

    return fit_dict

def get_chi_sq_fit() -> float:
    # retrieves the chi_sq value of the optimal fit from the fit file:

    chi_sq = None
    with open("fit_values.dat", 'r') as file:
        content = file.readlines()

        for line in content:
            if line.startswith("chi_sq:"):
                chi_sq = float(line[line.find("chi_sq: ") + 8:])

    return chi_sq


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
    asf = np.round(param_dict['asf'], 3)
    gep = np.round(param_dict['gep'], 1)
    gpp = np.round(param_dict['gpp'], 1)
    gamma = np.round(param_dict['gamma'], 1)

    file_name = f"a{asf}gep{gep}gpp{gpp}gamma{gamma}"

    return file_name

def assign_index_to(param: str) -> int:
    # returns a unique index for each parameter string
    index = None
    if param == "asf":
        index = 0
    elif param == "gep":
        index = 1
    elif param == "gpp":
        index = 2
    elif param == "gamma":
        index = 3
    elif param == "t0":
        index = 4
    elif param == "k":
        index = 5
    else:
        print(f"Param {param} not found in dict.")
        exit()

    return index

def assign_param_to(index: int) -> str:
    # inverse function to get_param_index()
    param = None
    if index == 0:
        param = "asf"
    elif index == 1:
        param = "gep"
    elif index == 2:
        param = "gpp"
    elif index == 3:
        param = "gamma"
    elif index == 4:
        param = "t0"
    elif index == 5:
        param = "k"
    else:
        print(f"Index {index} not found in list")
        exit()

    return param


def collect_all_func_vals(all_subsys: tuple, all_exp: dict, fit_dict: dict, file: str) -> NDArray:
    # returns the function values of the simulation data at the data points of experimental delays, appending all
    # datasets to one list

    cp_temp, cp_dat = (np.loadtxt('input_data/FGT/FGT_c_p1.txt')[:, 0], np.loadtxt('input_data/FGT/FGT_c_p1.txt')[:, 1])
    cp2_temp, cp2_dat = (np.loadtxt('input_data/FGT/FGT_c_p2.txt')[:, 0], np.loadtxt('input_data/FGT/FGT_c_p2.txt')[:, 1])
    all_func_vals = []
    for subsys in all_subsys:
        delay, sys_1, sys_2 = get_data(file=f"fits_global/{subsys}/{file}", subsys=subsys)

        if subsys == "el":
            delay += fit_dict["t0"] * 1e-3
            delay_indices = finderb(all_exp["el"][0], delay)
            all_func_vals += list(sys_1[delay_indices])
        elif subsys == "tp":
            delay_indices = finderb(all_exp["tp"][0], delay)
            temp_indices = finderb(sys_1, cp_temp)
            cp = cp_dat[temp_indices]
            ep = cp * sys_1

            temp2_indices = finderb(sys_2, cp2_temp)
            cp2 = cp2_dat[temp2_indices]
            ep2 = cp2 * sys_2

            ep_norm = (ep - ep[0]) / ep[finderb(10, delay)[0]] * fit_dict["k"]
            ep2_norm = (ep2 - ep2[0]) / ep2[finderb(10, delay)[0]] * (1 - fit_dict["k"])
            dat = ep_norm + ep2_norm / np.amax(ep_norm + ep2_norm)

            all_func_vals += list(dat[delay_indices])
        elif subsys == "mag":
            delay_indices = finderb(all_exp["mag"][0], delay)
            dat = 2 / 3 + 1 / 3 * sys_1
            all_func_vals += list(dat[delay_indices])

    all_func_vals = np.array(all_func_vals)
    return all_func_vals


def calculate_standard_deviations() -> None:
    # main function

    # static variables:
    all_subsys = ("el", "tp", "mag")
    all_exp = {"el": get_te_exp(), "tp": get_msd_exp(), "mag": get_mag_exp()}

    # get M optimal fit parameters, the corresponding file and value of chi_sq:
    opt_fit_dict = read_fit_file()
    opt_fit_file = get_filename_from_params(opt_fit_dict)
    opt_chi_sq = get_chi_sq_fit()
    M = len(opt_fit_dict)

    # get N function values of optimal fit at the N datapoints in experimental data:
    fit_func_vals = collect_all_func_vals(all_subsys=all_subsys, all_exp=all_exp, fit_dict= opt_fit_dict, file=opt_fit_file)
    N = len(fit_func_vals)

    # create Jacobi matrix of dimension NxM:
    J_ij = np.zeros((N, M))
    for param in opt_fit_dict.keys():
        j = assign_index_to(param)

        # vary parameters and get the corresponding N datapoints
        this_shift_dict_smaller = find_neighbouring_params(fit_dict=opt_fit_dict, param=param, smaller_or_bigger="smaller")
        this_shift_dict_bigger = find_neighbouring_params(fit_dict=opt_fit_dict, param=param, smaller_or_bigger="bigger")
        this_shift_file_smaller = get_filename_from_params(this_shift_dict_smaller)
        this_shift_file_bigger = get_filename_from_params(this_shift_dict_bigger)
        
        var_func_vals_smaller = collect_all_func_vals(all_subsys=all_subsys, all_exp=all_exp, fit_dict=this_shift_dict_smaller, file=this_shift_file_smaller)
        var_func_vals_bigger = collect_all_func_vals(all_subsys=all_subsys, all_exp=all_exp, fit_dict=this_shift_dict_bigger, file=this_shift_file_bigger)

        # calculate the derivative df/dp by linear approximation
        function_diff = var_func_vals_bigger - var_func_vals_smaller
        param_diff = this_shift_dict_bigger[param] - this_shift_dict_smaller[param]
        J_ij[:, j] = function_diff/param_diff
  
    print(opt_chi_sq/(N-M))
    # create the covariance matrix of dimension MxM:
    P_cov = opt_chi_sq/(N-M) * np.linalg.inv(np.matmul(J_ij.T, J_ij))

    # determine the standard deviation for each parameter and store in dictionary:
    sigmas = {}
    for j in range(len(P_cov)):
        param = assign_param_to(j)
        sigmas[param] = np.sqrt(P_cov[j, j])

    # print out results:
    print(f"Parameters: {opt_fit_dict} \n Sigmas: {sigmas}")

    return

if __name__ == "__main__":
    calculate_standard_deviations()
