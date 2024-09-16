import numpy as np
from scipy.integrate import solve_ivp
import os
import time
import warnings
from datetime import datetime

from .finderb import finderb

from .temps import SimTemperatures
from .temps import Sim11LL
from .temps import Sim11LD
from .temps import Sim11DD
from .temps import Sim12LL

from .mags import SimMagnetism
from .mags import SimNoMag

# supress printing of runtime warnings:
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SimDynamics:
    # This is the main Simulation class that holds the main function and unites all simulation steps
    def __init__(self, sample, pulse, end_time, ini_temp, solver, max_step, atol=1e-6, rtol=1e-3):
        # Input:
        # sample (object). The sample in use
        # pulse (object). The pulse excitation in use
        # end_time (float). Final time of simulation (including pulse delay) in s
        # ini_temp (float). Initial temperature of electron and phonon baths in the whole sample in K
        # constant_cp (boolean). If set True, cp_max is used for phonon heat capacity. If False, Einstein model is
        # computed. Note that this takes very long at low temperatures
        # solver (String). The solver used to evaluate the differential equation. See documentation of
        # scipy.integrate.solve_ivp
        # max_step (float). Maximum step size in s of the solver for the whole simulation
        # atol (float). Absolute tolerance of solve_ivp solver. Default is 1e-6 as the default of the solver.
        # rtol (float). Relative tolerance of solve_ivp solver. Default is 1e-3 as the default of the solver.

        # Also returns:
        # self.time_grid (numpy array). 1d-array of the time-grid to be used in simulations.

        self.Sam = sample
        self.Pulse = pulse
        self.end_time = end_time
        self.ini_temp = ini_temp
        self.time_grid = self.get_time_grid()
        self.solver = solver
        self.max_step = max_step
        self.atol = atol
        self.rtol = rtol

    def get_t_m_maps(self):
        # This method initiates all parameters needed for the dynamical simulation
        # and calls the solve_ivp function to run the sim.

        # Input:
        # self (object). The simulation object, taking sample and pulse as input

        # Returns:
        # all_sol (object). The solution and details of the simulation run by solve_ivp

        start_time = time.time()

        len_sam = self.Sam.len

        len_sam_te = self.Sam.len_te
        el_mask = self.Sam.el_mask
        mag_mask = self.Sam.mag_mask
        tp2_mask = self.Sam.tp2_mask
        el_mag_mask = self.Sam.el_mag_mask
        ce_gamma_sam = self.Sam.get_params('ce_gamma')[el_mask]
        mats, mat_ind = self.Sam.mats, self.Sam.mat_ind
        cp_sam_grid, cp_sam = self.Sam.get_params('cp_T')
        cp_sam = [np.array(i) for i in cp_sam]
        cp2_sam_grid, cp2_sam = self.Sam.get_params('cp2_T')
        cp2_sam = [np.array(i) for i in cp2_sam]
        gep_sam = self.Sam.get_params('gep')[el_mask]
        pulse_time_grid, pulse_map = self.Pulse.pulse_time_grid, self.Pulse.pulse_map
        dz_sam = self.Sam.get_params_from_blocks('dz')
        kappa_e_sam = self.Sam.get_params('kappae')
        kappa_p_sam = self.Sam.get_params('kappap')
        kappa_e_dz_pref = np.divide(kappa_e_sam, np.power(dz_sam, 2)[..., np.newaxis])[el_mask]
        kappa_p_dz_pref = np.divide(kappa_p_sam, np.power(dz_sam, 2)[..., np.newaxis])
        gpp_sam = self.Sam.get_params('gpp')
        j_sam = self.Sam.get_params('J')[mag_mask]
        spin_sam = self.Sam.get_params('spin')[mag_mask]
        arbsc_sam = self.Sam.get_params('arbsc')[mag_mask]
        s_up_eig_sq_sam = self.Sam.get_params('s_up_eig_squared')
        s_dn_eig_sq_sam = self.Sam.get_params('s_dn_eig_squared')
        mag_num = self.Sam.mag_num
        ms_sam = self.Sam.get_params('ms')
        vat_sam = self.Sam.get_params('vat')[mag_mask]

        print('++++++++++++++++++++++++++++ Simulation setup complete ++++++++++++++++++++++++++++')
        print()
        self.Sam.show_info()
        self.Pulse.show_info()
        print('Starting simulation')
        print()

        tem_mod = self.select_temp_dynamics()  # set things to None here
        mag_mod = self.select_mag_dynamics()  # here as well!

        te0, tp0 = self.initialize_temperature()  # different initializations with different sample setups!
        if mag_num != 0:
            fss0 = self.initialize_spin_configuration().flatten()

            fss_eq = SimMagnetism.equilibrate_mag(j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam,
                                                  fss0, te0, tp0[:len_sam][mag_mask], el_mag_mask, ms_sam, mag_num)
        else:
            fss_eq = np.zeros(1)

        print('Starting main simulation loop. Calculating until ', np.round((self.end_time-self.Pulse.delay)*1e12, 3),
              ' ps after pulse maximum.')
        print('Solver: ', str(self.solver))
        print('maximum time step:', str(self.max_step), ' s')
        print()

        ts = np.concatenate((te0, tp0))
        config0 = np.concatenate((ts, fss_eq))

        all_sol = solve_ivp(lambda t, all_baths: SimDynamics.get_t_m_increments(t, all_baths, len_sam, len_sam_te,
                                                                                mat_ind, el_mag_mask,
                                                                                mag_mask, el_mask, ce_gamma_sam,
                                                                                cp_sam_grid, cp_sam,
                                                                                gep_sam, pulse_map,
                                                                                pulse_time_grid, kappa_e_dz_pref,
                                                                                kappa_p_dz_pref, j_sam, spin_sam,
                                                                                arbsc_sam, s_up_eig_sq_sam,
                                                                                s_dn_eig_sq_sam, ms_sam, mag_num,
                                                                                vat_sam),
                            t_span=(0, self.time_grid[-1]), y0=config0, t_eval=self.time_grid, method=self.solver,
                            max_step=self.max_step)

        end_time = time.time()
        exp_time = end_time-start_time
        print('Simulation done. Time expired: ' + str(exp_time) + ' s')

        return all_sol

    def get_time_grid(self):
        # This method creates a time-grid for the simulation on the basis of the pulse-time-grid defined
        # in SimPulse class. Basically, after the pulse has been completely injected with a timestep of 0.1 fs,
        # use 10 fs as time-steps. This helps the solver as it does not have to recalculate new time-steps all the time.

        # Input:
        # self (object). The simulation to be run

        # Returns:
        # time_grid (numpy array). 1d-array of the time-steps to be used in the dynamical simulation

        start_time_grid = self.Pulse.pulse_time_grid
        if self.end_time < 5e-12:
            rest_time_grid = np.arange(start_time_grid[-1] + 1e-15, np.round(self.end_time, 15), 1e-15)
            time_grid = np.concatenate((start_time_grid, rest_time_grid))
        else:
            ep_time_grid = np.arange(start_time_grid[-1] + 1e-15, 5e-12, 1e-15)
            rest_time_grid = np.concatenate((ep_time_grid, np.arange(ep_time_grid[-1] + 1e-14, self.end_time, 1e-14)))
            time_grid = np.concatenate((start_time_grid, rest_time_grid))

        return time_grid

    def select_temp_dynamics(self):
        # This method checks which functions are needed to compute the temperature dynamics and assigns a proper
        # temperature object that bundles said functions in tm.temp_dyn().

        # Input:
        # self (Reference). Reference to the simulation object in use

        # Returns:
        # tem_mod (object). Temperature model that holds the necessary dynamical functions

        if self.Sam.len_tp2 == 0:
            if self.Sam.len == 1:
                tem_mod = Sim11LL()
            else:
                if self.Sam.len_te == 1:
                    tem_mod = Sim11LD()
                else:
                    tem_mod = Sim11DD()
        else:
            if self.Sam.len == 1 and self.Sam.len_te == 1:
                tem_mod = Sim12LL()
            else:
                print('There is no implemented method that fits the sample construction!')
                exit()

        return tem_mod

    def select_mag_dynamics(self):
        # This method checks which functions are needed to compute the temperature dynamics and assigns a proper
        # magnetization object that bundles said functions in tm.temp_dyn().

        # Input:
        # self (Reference). Reference to the simulation object in use

        # Returns:
        # mag_mod (object). Magnetization model that holds the necessary dynamical functions

        if self.Sam.mag_num == 0:
            mag_mod = SimNoMag()
        else:
            mag_mod = SimMagnetism()

        return mag_mod

    def initialize_temperature(self):
        # This method initializes the starting uniform temperature map.

        # Input:
        # self (object). The simulation to be run

        # Returns:
        # te_arr (numpy array). 1d-array of the starting electron temperatures
        # tp_arr (numpy array). 1d-array of the starting phonon temperatures

        te_arr = np.ones_like(self.Sam.mat_arr[self.Sam.el_mask])*self.ini_temp
        tp_arr = np.ones(self.Sam.len+self.Sam.len_tp2)*self.ini_temp

        return te_arr, tp_arr

    def initialize_spin_configuration(self):
        # This method initializes fully polarized spins in all magnetic layers (selected by whether mu_at > 0)
        # The maximal spin-level is being fully occupied, while the rest are empty.

        # Input:
        # self (object). The simulation to be run

        # Returns:
        # fss0 (numpy array). 1d-array of the initial spin configuration in reach magnetic layer

        fss0 = []

        for mat in self.Sam.mat_arr:
            if mat.muat != 0:
                fss0.append(np.zeros(int(2*mat.spin+1)))

        fss0 = np.array(fss0)
        fss0[:, 0] = 1

        return fss0

    @staticmethod
    def get_t_m_increments(timestep, te_tp_fs_flat, len_sam, len_sam_te, mat_ind, el_mag_mask,
                           mag_mask, el_mask, ce_gamma_sam,
                           cp_sam_grid, cp_sam, gep_sam, pulse_map, pulse_time_grid, kappa_e_dz_pref,
                           kappa_p_dz_pref, j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam,
                           ms_sam, mag_num, vat_sam):
        # This method joins other static methods to compute the increments of all three subsystems. It gets passed to
        # solve_ivp in the self.get_mag_map() method.

        # Input:
        # A bunch of parameters from the sample and pulse objects. See documentation in the respective methods

        # Returns:
        # all_increments_flat (numpy array). Flattened 1d-array of the increments of T_e, T_p
        # and fs (spin-level occupation in the magnetic material)
        te = te_tp_fs_flat[:len_sam_te]
        tp = te_tp_fs_flat[len_sam_te:len_sam_te+len_sam]

        if mag_num != 0:
            fss_flat = te_tp_fs_flat[len_sam_te+len_sam:]
            fss = np.reshape(fss_flat, (mag_num, (int(2 * spin_sam[0] + 1))))

            mag = SimMagnetism.get_mag(fss, ms_sam, spin_sam)
            dfs_dt = SimMagnetism.mag_occ_dyn(j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam,
                                              mag, fss, te, tp[mag_mask], el_mag_mask)
            dm_dt = SimMagnetism.get_mag(dfs_dt, ms_sam, spin_sam)

            mag_en_t = SimMagnetism.get_mag_en_incr(mag, dm_dt, j_sam, vat_sam)

            dfs_dt_flat = dfs_dt.flatten()
        else:
            mag_en_t = 0
            dfs_dt_flat = np.zeros(1)

        # compute local interactions of temperatures and pulse:
        cp_sam_t = np.zeros(len_sam)
        for i, ind_list in enumerate(mat_ind):
            cp_sam_grid_t = finderb(tp[ind_list], cp_sam_grid[i])
            cp_sam_t[ind_list] = cp_sam[i][cp_sam_grid_t]
        pulse_time = finderb(timestep, pulse_time_grid)[0]
        pulse_t = pulse_map[pulse_time][el_mask]
        ce_sam_t = np.multiply(ce_gamma_sam, te)

        dtp_dt = np.zeros(len_sam)

        dte_dt, dtp_dt[el_mask] = SimTemperatures.loc_temp_dyn(ce_sam_t, cp_sam_t[el_mask], gep_sam, te,
                                                               tp[el_mask], pulse_t, mag_en_t, el_mag_mask)

        # compute diffusion in temperature systems if the sample contains more than one layer:
        if len(te) == 1:
            dte_dt_diff = np.zeros(1)
        else:
            dte_dt_diff = SimTemperatures.electron_diffusion(kappa_e_dz_pref, ce_sam_t, te, tp[el_mask])
        if len(tp) == 1:
            dtp_dt_diff = np.zeros(1)
        else:
            dtp_dt_diff = SimTemperatures.phonon_diffusion(kappa_p_dz_pref, cp_sam_t, tp)

        dte_dt += dte_dt_diff
        dtp_dt += dtp_dt_diff

        # bring data of increments back into 1d array shape:
        dtep_dt = np.concatenate((dte_dt, dtp_dt))
        all_increments_flat = np.concatenate((dtep_dt, dfs_dt_flat))

        return all_increments_flat

    def separate_data(self, sim_results):
        # This method takes the 2-dimensional output of the solver and separates it accordingly to the
        # given sample structure into, te, tp and mag maps.

        # Input:
        # self (object). The simulation you just ran
        # sim_results (object). The output of the solve_ivp function.

        # Returns:
        # sim_delay (numpy array). 1d-array of the time steps used in the solve_ivp function
        # tes (numpy array). 2d-array of the electron temperature map of dimension len(sim_delay) x self.Sam.len
        # tps (numpy array). 2d-array of the phonon temperature map of dimension len(sim_delay) x self.Sam.len
        # mags (numpy array). 2d-array of the magnetization map of dimension len(sim_delay) x self.Sam.mag_num

        sim_delay = sim_results.t
        sim_results = sim_results.y.T
        tes = sim_results[:, :self.Sam.len_te]
        tps = sim_results[:, self.Sam.len_te:self.Sam.len_te + self.Sam.len]

        if self.Sam.mag_num != 0:
            fss_flat = sim_results[:, self.Sam.len_te + self.Sam.len:]
            fss = np.reshape(fss_flat, (len(sim_delay), self.Sam.mag_num,
                             int(2*self.Sam.get_params('spin')[self.Sam.mag_mask][0]+1)))
            mags = self.get_mag_results(fss)

        else:
            mags = np.zeros(1)

        return sim_delay, tes, tps, mags

    def get_mag_results(self, fss):
        # This method takes the map of spin occupations fs and converts it into the magnetization map.

        # Input:
        # fss (numpy array). 3d-array of the spin-level occupations of dimension
        # len(sim_delay) x len(self.Sam.mag_num) x (2 * effective_spin_of_magnetic_material + 1)

        # Returns:
        # mags (numpy array). 2d-array of the magnetization map of dimension len(sim_delay) x len(self.Sam.mag_num)

        mags = -np.divide(np.sum(self.Sam.get_params('ms')[np.newaxis, ...] * fss, axis=-1),
                          self.Sam.get_params('spin')[self.Sam.mag_mask])

        return mags

    def save_data(self, sim_results, save_file):
        # This method saves the simulation results in the 'Results' folder in a desired filename. It calls the
        # two functions above to convert the data properly.

        # Input:
        # self (object). The simulation just run.
        # sim_results (object). The output of the solve_ivp function.
        # save_file (string). Desired filename for saving the data.

        # Returns:
        # None. After saving the data in the proper format and folder, the method closes without output.

        sim_path = 'Results/' + str(save_file)

        sim_delay, sim_tes, sim_tps, sim_mags = self.separate_data(sim_results)

        if not os.path.exists(sim_path):
            os.makedirs(sim_path)
        for file in os.listdir(sim_path):
            os.remove(os.path.join(sim_path, file))

        np.save(sim_path + '/tes.npy', sim_tes)
        np.save(sim_path + '/tps.npy', sim_tps)
        np.save(sim_path + '/ms.npy', sim_mags)
        np.save(sim_path + '/delay.npy', sim_delay)

        mats = self.Sam.mats

        params_file = open(sim_path + '/params.dat', 'w+')

        params_file.write('##Simulation parameters' + '\n')
        params_file.write('Time of finalization: ' + str(datetime.now()) + ' GMT' + '\n')
        params_file.write('Integration method: ' + self.solver + '\n')
        params_file.write('Relative tolerance of solver: ' + str(self.rtol) + '\n')
        params_file.write('Absolute tolerance of solver: ' + str(self.atol) + '\n')
        params_file.write('Maximum timestep of integration: ' + str(self.max_step) + ' s' +'\n')
        params_file.write('Initial temperature: ' + str(self.ini_temp) + ' [K]' + '\n')
        params_file.write('Simulation time span: ' + str(self.end_time*1e9) + ' [ns]' + '\n')
        params_file.write('##Sample parameters' + '\n')
        params_file.write('Materials: ' + str([mat.name for mat in mats]) + '\n')
        params_file.write('Constituents: ' + str([mat for mat in self.Sam.constituents]) + '\n')
        params_file.write('Material positions at layer in sample: ' + str(self.Sam.mat_ind) + '\n')
        params_file.write('Layer depth of constituents = ' + str(list(self.Sam.dz_arr)) + ' [m]' + '\n')
        params_file.write('Atomic volumes = ' + str([mat.vat for mat in mats]) + ' [m^3]' + '\n')
        params_file.write('Effective spin = ' + str([mat.spin for mat in mats]) + '\n')
        params_file.write('mu_at = ' + str([mat.muat for mat in mats]) + ' [mu_Bohr]' + '\n')
        params_file.write('a_sf = ' + str([mat.asf for mat in mats]) + '\n')
        params_file.write('R = ' + str([mat.R*1e-12 for mat in mats]) + ' [1/ps]' + '\n')
        params_file.write('g_ep = ' + str([mat.gep for mat in mats]) + ' [W/m^3/K]' + '\n')
        params_file.write('gamma_el = ' + str([mat.ce_gamma for mat in mats]) + ' [J/m^3/K^2]' + '\n')
        params_file.write('cv_ph_max = ' + str([mat.cp_max for mat in mats]) + ' [J/m^3/K]' + '\n')
        params_file.write('kappa_el = ' + str([mat.kappae for mat in mats]) + ' [W/mK]' + '\n')
        params_file.write('kappa_ph = ' + str([mat.kappap for mat in mats]) + ' [W/mK]' + '\n')
        params_file.write('T_C = ' + str([mat.tc for mat in mats]) + ' [K]' + '\n')
        params_file.write('T_Deb = ' + str([mat.tdeb for mat in mats]) + ' [K]' + '\n')
        params_file.write('##Pulse parameters' + '\n')
        if self.Pulse.method == 'LB':
            params_file.write('Absorption profile computed with Lambert-Beer-Law' + '\n')
            params_file.write('Penetration depths: ' + str(self.Sam.pen_dep_arr*1e9) + ' nm' + '\n')
        elif self.Pulse.method == 'Abeles':
            params_file.write('Absorption profile computed with Abeles\' matrix method' + '\n')
            params_file.write('Refractive indices: ' + str(self.Sam.n_comp_arr) + '\n')
            params_file.write('Photon energy: ' + str(self.Pulse.energy) + ' eV' +'\n')
            params_file.write('Incident angle: ' + str(self.Pulse.theta/np.pi) + ' pi' + '\n')
            params_file.write('Polarization angle: ' + str(self.Pulse.phi/np.pi) + ' pi' +'\n')
        params_file.write('Sigma = ' + str(self.Pulse.pulse_width) + ' [s]' + '\n')
        params_file.write('Delay = ' + str(self.Pulse.delay) + ' [s]' + '\n')
        params_file.write('Incident fluence = ' + str(self.Pulse.fluence) + ' [mJ/cm^2]' + '\n')
        params_file.write('Absorbed fluence = ' + str(self.Pulse.abs_flu) + ' [mJ/cm^2]' + '\n')
        params_file.write('Reflected fluence = ' + str(self.Pulse.ref_flu) + ' [mJ/cm^2]' + '\n')
        params_file.write('Transmitted fluence = ' + str(self.Pulse.trans_flu) + ' [mJ/cm^2]' + '\n')
        params_file.write('Absorbed fluence per block of different materials: ' + str(self.Pulse.abs_flu_per_block)
                          + ' [mJ/cm^2]' + '\n')
        params_file.write('Relative error due to finite layer size: ' + str(self.Pulse.rel_err) + ' %' + '\n')
        params_file.write('##Interface parameters' + '\n')
        params_file.write('kappa_e_int = ' + str(self.Sam.kappa_e_int/self.Sam.dz_arr[1:]) + ' [MW/m^2/K]' + '\n')
        params_file.write('kappa_p_int = ' + str(self.Sam.kappa_p_int/self.Sam.dz_arr[1:]) + ' [MW/m^2/K]' + '\n')
        params_file.close()

        return
