import numpy as np
import os
from scipy import io
from scipy.stats import chi2
import time
import multiprocessing as mp

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


def compute_chi_sq_for_gamma(gamma_index, gamma, files_te, files_mag, files_tp, folder_str, t0_el, geps, asfs, gpps, k_ep, cp_data, cp2_data, exp_data):
    chi_sq_loc = np.zeros((20, 16, 11, 16, 20), dtype=float)  # t0, gep, asf, gpp, k

    # Load experimental and simulation data
    temp, cp_temp = cp_data
    temp2, cp2_temp = cp2_data
    (exp_delay_mag, exp_dat_mag, sigma_rel_mag,
     exp_delay_te, exp_dat_te, sigma_rel_te,
     exp_delay_tp, exp_dat_tp, sigma_rel_tp) = exp_data

    # Loop through subsystems
    for folder, f_str in zip([files_te, files_mag, files_tp], folder_str):
        for file in folder:
        
            if not file.startswith("a") or gamma != float(file[file.find("gamma")+5:]):
                continue

            # Extract parameters from the filename
            asf = float(file[file.find('a') + 1: file.find("gep")])
            gep = float(file[file.find('gep') + 3: file.find("gpp")])
            gpp = float(file[file.find("gpp") + 3: file.find("gamma")])

            asf_index = finderb(asf, asfs)[0]
            gep_index = finderb(gep, geps)[0]
            gpp_index = finderb(gpp, gpps)[0]

            full_path = f'fits_global/{f_str}{file}'

            # depending on the subsystem, the fit will be done differently, for te and tp one needs extra loops:
            if f_str == 'mag/':
                delay, dat, _ = get_data(full_path, 'mag')
                dat = 2 / 3 + 1 / 3 * dat
                exp_delay, exp_dat, sigma_rel = exp_delay_mag, exp_dat_mag, sigma_rel_mag

                delay_indices = finderb(exp_delay, delay)
                cs_norm = np.sum(((exp_dat - dat[delay_indices]) / sigma_rel) ** 2)
                chi_sq_loc[:, gep_index, asf_index, gpp_index, :] += cs_norm

            elif f_str == 'el/':
                delay, dat, _ = get_data(full_path, 'te')
                exp_delay, exp_dat, sigma_rel = exp_delay_te, exp_dat_te, sigma_rel_te

                for t0_index, t0_shift in enumerate(t0_el):
                    delay += t0_shift * 1e-13
                    delay_indices = finderb(exp_delay, delay)
                    cs_norm = np.sum(((exp_dat - dat[delay_indices]) / sigma_rel) ** 2)
                    chi_sq_loc[t0_index, gep_index, asf_index, gpp_index, :] += cs_norm

            elif f_str == 'tp/':
                delay, tp, tp2 = get_data(full_path, 'tp')
                exp_delay, exp_dat, sigma_rel = exp_delay_tp, exp_dat_tp, sigma_rel_tp

                temp_indices = finderb(tp, temp)
                cp = cp_temp[temp_indices]
                ep = cp * tp

                temp2_indices = finderb(tp2, temp2)
                cp2 = cp2_temp[temp2_indices]
                ep2 = cp2 * tp2

                delay_indices = finderb(exp_delay, delay)

                for k_index, k in enumerate(k_ep):
                    # set maximum at 10 ps (a little arbitrary I must confess)
                    ep_norm = (ep - ep[0]) / ep[finderb(10, delay)[0]] * k
                    ep2_norm = (ep2 - ep2[0]) / ep2[finderb(10, delay)[0]] * (1 - k)

                    dat = ep_norm + ep2_norm / np.amax(ep_norm + ep2_norm)

                    cs_norm = np.sum(((exp_dat - dat[delay_indices]) / sigma_rel) ** 2)
                    chi_sq_loc[:, gep_index, asf_index, gpp_index, k_index] += cs_norm

            else:
                print(f"File could not be assigned to subsystem: {f_str}{file}")

    # find the best local fit for each gamma and return:
    best_chi_sq_loc = np.amin(chi_sq_loc)
    min_ind = np.argmin(chi_sq_loc)
    t0_id_loc, gep_id_loc, asf_id_loc, gpp_id_loc, k_id_loc = np.unravel_index(min_ind, chi_sq_loc.shape)

    gamma_loc = gamma
    t0_loc = t0_el[t0_id_loc]
    gep_loc = geps[gep_id_loc]
    asf_loc = asfs[asf_id_loc]
    gpp_loc = gpps[gpp_id_loc]

    best_params = (gamma_loc, t0_loc, gep_loc, asf_loc, gpp_loc)
    return best_chi_sq_loc, best_params


def global_manual_fit_parallel():
    start = time.time()

    # Define parameter ranges
    gammas = np.arange(205, 226).astype(float)  # 21 values
    t0_el = np.arange(20)  # 20 values
    geps = np.arange(4.0, 5.6, 0.1)  # 16 values
    asfs = np.arange(0.01, 0.021, 0.001)  # 11 values
    gpps = np.arange(2.5, 4.1, 0.1)  # 16 values
    k_ep = np.arange(5, 25) * 2e-2  # 20 values

    # Prepare file lists and folder structure
    files_te = os.listdir('Results/FGT/fits_global/el')
    files_mag = os.listdir('Results/FGT/fits_global/mag')
    files_tp = os.listdir('Results/FGT/fits_global/tp')
    folder_str = ['el/', 'mag/', 'tp/']

    # Load phonon heat capacity data
    cp_data = (np.loadtxt('input_data/FGT/FGT_c_p1.txt')[:, 0], np.loadtxt('input_data/FGT/FGT_c_p1.txt')[:, 1])
    cp2_data = (np.loadtxt('input_data/FGT/FGT_c_p2.txt')[:, 0], np.loadtxt('input_data/FGT/FGT_c_p2.txt')[:, 1])

    # Load experimental data
    exp_data = (
        *get_mag_exp(),  # exp_delay_mag, exp_dat_mag, exp_dd_mag, sigma_rel_mag
        *get_te_exp(),   # exp_delay_te, exp_dat_te, exp_dd_te, sigma_rel_te
        *get_msd_exp()   # exp_delay_tp, exp_dat_tp, exp_dd_tp, sigma_rel_tp
    )

    # Prepare multiprocessing pool
    with mp.Pool(processes=21) as pool:
        results = pool.starmap(
            compute_chi_sq_for_gamma,
            [(gamma_index, gamma, files_te, files_mag, files_tp, folder_str, t0_el, geps, asfs, gpps, k_ep, cp_data, cp2_data, exp_data)
             for gamma_index, gamma in enumerate(gammas)]
        )

    # Aggregate results
    best_chi_sq = float('inf')
    best_params = None
    for chi_sq, params in results:
        if chi_sq < best_chi_sq:
            best_chi_sq = chi_sq
            best_params = params

    # Extract best-fit parameters from indices
    gamma_fit, t0_fit, gep_fit, asf_fit, gpp_fit, k_fit = best_params

    runtime = time.time() - start

    # Save results
    with open("fit_values.dat", 'w+') as file:
        file.write(f"gamma: {gamma_fit}\n")
        file.write(f"t0: {t0_fit}\n")
        file.write(f"gep: {gep_fit}\n")
        file.write(f"asf: {asf_fit}\n")
        file.write(f"gpp: {gpp_fit}\n")
        file.write(f"k: {k_fit}\n")
        file.write(f"time spent: {runtime}\n")

    return


if __name__ == "__main__":
    global_manual_fit_parallel()
