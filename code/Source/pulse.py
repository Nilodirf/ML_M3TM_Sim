import numpy as np
import scipy
import os
from matplotlib import pyplot as plt
from scipy import constants as sp

from ..Source.finderb import finderb


class SimPulse:
    # This class lets us define te pulse for excitation of the sample

    def __init__(self, sample, pulse_width, fluence, delay, method, photon_energy_ev=None, theta=None, phi=None):
        # Input:
        # sample (object). Sample in use
        # pulse_width (float). Sigma of gaussian pulse shape in s
        # fluence (float). Fluence of the laser pulse in mJ/cm**2. Converted to J/m**2
        # delay (float). time-delay of the pulse peak after simulation start in s (Let the magnetization relax
        # to equilibrium before injecting the pulse
        # method (String). The method to calculate the pulse excitation map. Either 'LB' for Lambert-Beer or 'Abele'
        # for the matrix formulation calculating the profile via the Fresnel equations.
        # energy (float). Energy of the optical laser pulse in eV. Only necessary for method 'Abele'
        # theta (float). Angle of incidence of the pump pulse in respect to the sample plane normal in units of pi, so
        # between 0 and 1/2.
        # phi (float). Angle of polarized E-field of optical pulse in respect to incidence plane in units of pi, so
        # between 0 and 1/2.

        # Also returns:
        # peak_intensity (float). Peak power per area (intensity) of the pulse in W/m**2.
        # Needed to compute the absorption profile
        # pulse_time_grid, pulse_map (numpy arrays). 1d-arrays of the time-grid on which the pulse is defined
        # and the corresponding 2d-array of excitation power density at all times in all layers
        # abs_flu (float). The absorbed fluence calculated for the sample structure, without errors for
        # finite timesteps.
        # ref_flu (float). The reflected fluence calculated for the sample structure, without timestep errors
        # rel_err (float). Error for absorbed fluence due to finite layer size in percent, rounded to 0.1 %


        self.pulse_width = pulse_width
        self.fluence = fluence
        self.delay = delay
        self.peak_intensity = self.fluence/np.sqrt(2*np.pi)/self.pulse_width*10
        self.Sam = sample
        self.method = method
        assert self.method == 'LB' or self.method == 'Abeles', 'Chose one of the methods \'LB\' (for Lambert-Beer)' \
                                                              ' or \' Abeles\' (for computation of Fresnel equations).'
        self.energy = photon_energy_ev
        self.theta = np.pi * theta if theta is not None else None
        self.phi = np.pi * phi if phi is not None else None
        self.pulse_time_grid, self.pulse_map, self.abs_flu, self.ref_flu, self.trans_flu, self.rel_err = self.get_pulse_map()
        
    def get_for_all_layers(self, array):
        # This method takes 1d array defined on N blocks of the sample and returns the data stored in this array for all
        # layers in the sample.

        # Input:
        # self (object). The pulse in use
        # array (numpy array). The 1d-array that shall be extended

        # Returns:
        # array_for_layers (numpy array). The 1d-array for all layers in the sample

        return np.concatenate(np.array([[array[i] for _ in range(self.Sam.mat_blocks[i])] for i in range(len(self.Sam.mat_blocks))], dtype=object)).astype(complex)

    def get_pulse_map(self):
        # This method creates a time grid and a spatially independent pulse excitation of gaussian shape
        # on this time grid.
        # The pulse is confined to nonzero values in the range of [start_pump_time, end_pump_time]
        # to save computation time. After this time, there follows one entry defining zero pump power
        # for all later times. At each timestep in the grid, the spatial dependence of the pulse is multiplied
        # via self.depth_profile

        # Input:
        # self (object). The pulse defined by the parameters above

        # Returns:
        # pump_time_grid (numpy array). 1d-array of the time grid on which the pulse is defined
        # pump_map (numpy array). 2d-array of the corresponding pump energies on the time grid (first dimension)
        # and for the whole sample (second dimension)
        p_del = self.delay
        sigma = self.pulse_width
        start_pump_time = p_del-10*sigma
        end_pump_time = p_del+10*sigma

        raw_pump_time_grid = np.arange(start_pump_time, end_pump_time, 1e-16)
        until_pump_start_time = np.arange(0, start_pump_time, 1e-16)
        pump_time_grid = np.append(until_pump_start_time, raw_pump_time_grid)

        raw_pump_grid = np.exp(-((raw_pump_time_grid-p_del)/sigma)**2/2)
        pump_grid = np.append(np.zeros_like(until_pump_start_time), raw_pump_grid)

        pump_time_grid = np.append(pump_time_grid, end_pump_time+1e-15)
        pump_grid = np.append(pump_grid, 0.)

        pump_map, abs_flu, ref_flu, trans_flu, rel_err = self.depth_profile(pump_grid)

        return pump_time_grid, pump_map, abs_flu, ref_flu, trans_flu, rel_err

    def depth_profile(self, pump_grid):
        # This method computes the depth dependence of the laser pulse. Either from Lambert-Beer law or from Abeles'
        # matrix method.

        # Input:
        # sample (class object). The before constructed sample
        # pump_grid (numpy array). 1d-array of timely pulse shape on the time grid defined in create_pulse_map

        # Returns:
        # 2d-array of the corresponding pump energies on the time grid (first dimension)
        # and for the whole sample (second dimension)

        dz_sam = self.Sam.get_params_from_blocks('dz')
        mat_blocks = self.Sam.mat_blocks

        max_intensity = self.peak_intensity
        powers = np.array([])

        if self.method == 'LB':

            pendep_sam = self.Sam.get_params_from_blocks('pen_dep')
            assert pendep_sam.any() is not None, 'Define penetration depths for all blocks of the sample within the' \
                                                 'SimSample.add_layers() method when choosing Lambert-Beer absorption '\
                                                 'profile.'

            first_layer = 0
            last_layer = 0

            already_penetrated = 0

            for i in range(len(mat_blocks)):
                last_layer += mat_blocks[i]
                if pendep_sam[first_layer] == 1:
                    powers = np.append(powers, np.zeros(mat_blocks[i]))
                    first_layer = last_layer
                    continue
                pen_red = np.divide((np.arange(mat_blocks[i])+already_penetrated)*dz_sam[first_layer:last_layer],
                                    pendep_sam[first_layer:last_layer]).astype(float)
                powers = np.append(powers, max_intensity/pendep_sam[first_layer:last_layer]
                                   * np.exp(-pen_red))
                max_intensity = powers[-1]*pendep_sam[last_layer-1]
                first_layer = last_layer
                already_penetrated = 1
            abs_flu = np.sum(powers*dz_sam) * (np.sqrt(2*np.pi)*self.pulse_width*10)
            trans_flu = self.fluence-abs_flu
            ref_flu = 0
            rel_err = None
            excitation_map = np.multiply(pump_grid[..., np.newaxis], np.array(powers))

        elif self.method == 'Abeles':

            assert self.Sam.n_comp_arr.any() is not None, 'Please define a refractive index for every constituent'\
                                                               'of the sample within the definition of the sample.'
            assert self.energy is not None and self.theta is not None and self.phi is not None, \
                'For the chosen method, make sure photon energy, theta and phi are defined.'

            # N is the number of blocks/constituents in the sample, so all in all we have N+2 blocks in the system,
            # considering vacuum before and after the sample:
            N = len(self.Sam.mat_blocks)

            # wavelength in m of laser pulse from photon energy:
            wave_length = sp.h*sp.c/sp.physical_constants['electron volt'][0]/self.energy

            # compute the normalized electric field amplitudes of p/s waves from the given angle phi:
            e_p0 = np.cos(self.phi)
            e_s0 = np.sin(self.phi)

            # set up array of refraction indices, first layer and last layer considered vacuum before/after sample:
            n_comp_arr = np.append(np.append(np.ones(1, dtype=complex), self.Sam.n_comp_arr), np.ones(1, dtype=complex))

            # compute the penetration angle theta in every sample constituent from Snell's law:
            theta_arr = np.empty(N+2, dtype=complex)
            theta_arr[0] = self.theta
            theta_arr[1:] = np.arcsin(n_comp_arr[0]/n_comp_arr[1:]*np.sin(theta_arr[0]))

            # fresnel equations at N+1 interfaces:
            n_last = n_comp_arr[:-1]
            n_next = n_comp_arr[1:]
            cos_theta_last = np.cos(theta_arr[:-1])
            cos_theta_next = np.cos(theta_arr[1:])

            r_s = np.divide(n_last*cos_theta_last-n_next*cos_theta_next, n_last*cos_theta_last+n_next*cos_theta_next)
            t_s = np.divide(2*n_last*cos_theta_last, n_last*cos_theta_last+n_next*cos_theta_next)
            r_p = np.divide(n_last*cos_theta_next-n_next*cos_theta_last, n_last*cos_theta_next+n_next*cos_theta_last)
            t_p = np.divide(2*n_last*cos_theta_last, n_last*cos_theta_next+n_next*cos_theta_last)

            # we need the thicknesses of blocks and the distance from previous interfaces, for the N inner blocks:
            penetration_from_interface = np.array([])
            block_thicknesses = np.array([])
            start = 0
            for end in self.Sam.mat_blocks:
                end += start
                penetration_from_interface = np.append(penetration_from_interface, np.cumsum(dz_sam[start:end])-dz_sam[start])
                block_thicknesses = np.append(block_thicknesses, np.sum(dz_sam[start:end]))
                start = end

            # now the propagation matrices, for N+1 blocks, the first vacuum layer cannot be defined in this sense:
            all_C_s_mat = np.empty((N+1, 2, 2), dtype=complex)
            all_C_p_mat = np.empty((N+1, 2, 2), dtype=complex)

            all_C_s_mat[0] = np.array([[1, r_s[0]], [r_s[0], 1]])  # 0 corresponds to C_s(p)_1 here
            all_C_p_mat[0] = np.array([[1, r_p[0]], [r_p[0], 1]])

            # N phases for the N blocks of the sample
            all_phases = 1j*2*np.pi/wave_length*n_comp_arr[1:-1]*cos_theta_last[1:]*block_thicknesses

            for i in range(1, len(all_C_s_mat)):
                # starts at i=1, so after the first sample block, C_s(p)_2
                all_C_s_mat[i] = np.array([[np.exp(-all_phases[i-1]), r_s[i]*np.exp(-all_phases[i-1])],
                                          [r_s[i]*np.exp(all_phases[i-1]), np.exp(all_phases[i-1])]])
                all_C_p_mat[i] = np.array([[np.exp(-all_phases[i-1]), r_p[i]*np.exp(-all_phases[i-1])],
                                           [r_p[i]*np.exp(all_phases[i-1]), np.exp(all_phases[i-1])]])

            # now D_matrices for all N+2 blocks:
            all_D_s_mat = np.empty((N+2, 2, 2), dtype=complex)
            all_D_p_mat = np.empty((N+2, 2, 2), dtype=complex)

            # fill last one in vacuum with identity:
            all_D_s_mat[-1] = np.identity(2)
            all_D_p_mat[-1] = np.identity(2)

            for i in range(len(all_D_p_mat)-2, -1, -1):
                # starts in last block of sample (i=N), loops until first vacuum (i=0)
                all_D_s_mat[i] = np.matmul(all_C_s_mat[i], all_D_s_mat[i+1])
                all_D_p_mat[i] = np.matmul(all_C_p_mat[i], all_D_p_mat[i+1])

            # total reflection, transmission:
            t_p_tot = np.real(np.divide(cos_theta_next[-1], cos_theta_last[0]))\
                      * np.abs(np.prod(t_p)/all_D_p_mat[0, 0, 0])**2
            t_s_tot = np.real(np.divide(cos_theta_next[-1], cos_theta_last[0]))\
                      * np.abs(np.prod(t_s)/all_D_s_mat[0, 0, 0])**2
            r_s_tot = np.abs(all_D_s_mat[0, 1, 0] / all_D_s_mat[0, 0, 0])**2
            r_p_tot = np.abs(all_D_p_mat[0, 1, 0] / all_D_p_mat[0, 0, 0])**2

            # electric field in all N+1 blocks (excluding the first vacuum),
            # for +(-) propagating waves of layer j indexed [j,0(1)]:
            all_E_s_amps = np.empty((N+1, 2), dtype=complex)
            all_E_p_amps = np.empty((N+1, 2), dtype=complex)

            for i in range(N+1):
                # i=0 corresponds to the first sample block, so E_1
                tp = np.prod(t_p[:i+1])
                ts = np.prod(t_s[:i+1])
                t_p_ep = tp * e_p0
                t_s_es = ts * e_s0
                all_E_p_amps[i, 0] = all_D_p_mat[i+1, 0, 0]/all_D_p_mat[0, 0, 0]*t_p_ep
                all_E_p_amps[i, 1] = all_D_p_mat[i+1, 1, 0]/all_D_p_mat[0, 0, 0]*t_p_ep
                all_E_s_amps[i, 0] = all_D_s_mat[i+1, 0, 0]/all_D_s_mat[0, 0, 0]*t_s_es
                all_E_s_amps[i, 1] = all_D_s_mat[i+1, 1, 0]/all_D_s_mat[0, 0, 0]*t_s_es

            # kz in all N blocks of the sample:
            kz_in_sample = 2*np.pi/wave_length*n_comp_arr[1:-1]*np.cos(theta_arr[1:-1])

            # electric field in all layers of sample and proportionality factor for absorption:
            e_p_in_sample = np.empty((self.Sam.len, 2), dtype=complex)
            e_s_in_sample = np.empty((self.Sam.len, 2), dtype=complex)
            q_prop = np.empty(self.Sam.len)

            first_layer = 0
            for i, last_layer in enumerate(self.Sam.mat_blocks):
                # i=0 corresponds to the first sample block here, so E_1
                last_layer += first_layer
                phase = np.array(1j*kz_in_sample[i]*penetration_from_interface[first_layer:last_layer], dtype=complex)
                phase = np.array([np.exp(phase), np.exp(-phase)]).T
                e_s_in_sample[first_layer: last_layer] = phase * all_E_s_amps[i, :]
                e_p_in_sample[first_layer: last_layer] = phase * all_E_p_amps[i, :]
                q_prop[first_layer: last_layer] = np.real(np.divide(n_comp_arr[i+1]*np.cos(theta_arr[i+1]), np.cos(self.theta)))\
                                                  * 2*np.imag(kz_in_sample[i])
                first_layer = last_layer

            e_x_in_sample = np.sum(e_p_in_sample, axis=-1) * np.cos(self.get_for_all_layers(theta_arr[1:-1]))
            e_z_in_sample = np.concatenate(np.diff(e_p_in_sample, axis=-1)) * np.sin(self.get_for_all_layers(theta_arr[1:-1]))
            e_y_in_sample = np.sum(e_s_in_sample, axis=-1)

            # from the E-field we get the normalized intensity:
            F_z_p = (np.abs(e_x_in_sample)**2 + np.abs(e_z_in_sample)**2) if self.phi != np.pi*1/2 else 0
            F_z_s = np.abs(e_y_in_sample)**2 if self.phi != 0 else 0
            F_z = F_z_p + F_z_s

            # and finally the absorbed power densities:
            powers = self.peak_intensity*q_prop*F_z
            abs_flu = self.fluence * np.sum(F_z * dz_sam*q_prop)
            ref_flu = self.fluence * (e_p0**2 * r_p_tot + e_s0**2 * r_s_tot)
            trans_flu = self.fluence * (e_p0**2 * t_p_tot + e_s0**2 * t_s_tot)
            rel_err = np.round(100*(abs_flu-(self.fluence-trans_flu-ref_flu))/abs_flu, 2)

            excitation_map = np.multiply(pump_grid[..., np.newaxis], powers)

        return excitation_map.astype(float), abs_flu, ref_flu, trans_flu, rel_err

    def visualize(self, axis, fit=None, save_fig=False, save_file=None):
        # This method plots the spatial/temporal/both dependencies of the pump pulse.

        # Input:
        # self (object). The pulse object in use
        # axis (String). Chose whether to plot the temporal profile ('t'), the spatial profile of absorption ('z')
        # or both ('tz')
        # save_fig (boolean). Wheather to save the figure of the absorption profile
        # save_file (string). Name of the file to save pulse absorption visualization if save_fig=True

        # Returns:
        # None. It is void function that only plots.

        assert axis == 't' or axis == 'z' or axis == 'tz', 'Please select one of the three plotting options \'t\' ' \
                                                        '(to see time dependence), \'z\' (to see depth dependence),' \
                                                        '\'tz\' (to see the pulse map in time and depth).'
        plt.figure(figsize=(8, 6))

        if axis == 't':
            norm = np.amax(self.pulse_map)
            j = 0
            max = 0.
            for layer_index in range(self.Sam.len):
                if self.pulse_map[:, layer_index].any() > max:
                    max = np.amax(self.pulse_map[:, layer_index])
                    j = layer_index
            plt.plot(self.pulse_time_grid, self.pulse_map[:, j]/norm)
            plt.xlabel(r'delay [ps]', fontsize=16)
            plt.ylabel(r'S(t)/S$_{max}$', fontsize=16)
            plt.show()

        elif axis == 'z':
            dz_sam = self.Sam.get_params_from_blocks('dz')
            norm = np.amax(self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :])
            # norm = 1
            sample_depth = np.cumsum(dz_sam) - dz_sam[0]
            powers_to_plot = self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :]/norm
            plt.plot(sample_depth*1e9, powers_to_plot, color='black', lw=2.0)
            plt.ylim(0, np.amax(powers_to_plot))
            plt.xlim(0, sample_depth[-1]*1e9)
            plt.xlabel(r'sample depth z [nm]', fontsize=16)
            plt.ylabel(r'Norm. absorbed energy', fontsize=16)

            # add fillings to denote different sample constituents:
            top_fill = np.amax(powers_to_plot)
            depth_0 = 0.
            for i, mat in enumerate(self.Sam.constituents):
                depth_range = np.array([depth_0, depth_0 + self.Sam.dz_arr[i] * 1e9 * self.Sam.mat_blocks[i]])
                plt.fill_between(depth_range, np.ones(2)*top_fill, np.zeros(2), alpha=0.5, label=str(mat))
                depth_0 = depth_range[-1]
            if self.method == 'LB':
                plt.title('Absorption profile computed with Lambert-Beer law', fontsize=20)
            if self.method == 'Abeles':
                plt.title('Absorption profile computed with Abeles matrix method', fontsize=20)
            plt.legend(fontsize=16)

            if save_fig:

                assert save_file is not None, 'If you wish to save, please introduce a name for the file with save_file=\'name\''

                save_path = 'Results/' + str(save_file) + '.pdf'

                if not os.path.exists('Results'):
                    os.makedirs('Results')

                plt.savefig('Results/' + str(save_file) + '.pdf')

            plt.show()

            if fit is not None:
                if fit == 'exp':
                    def fit_func(depth, pen_dep):
                        return np.exp(-depth/pen_dep)
                elif fit == 'lin':
                    def fit_func(depth, slope):
                        return 1-(slope*depth)
                excited_depth = sample_depth[self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :] != 0].astype(float)
                excited_depth -= excited_depth[0]
                to_fit = self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :] / norm
                to_fit = to_fit[to_fit != 0]
                p0 = 30e-9
                pen_dep, cv = scipy.optimize.curve_fit(fit_func, excited_depth, to_fit, p0)
                plt.plot(excited_depth, to_fit, ls='dotted', lw=3, label='Abeles\' method')
                plt.plot(excited_depth, fit_func(excited_depth, pen_dep[0]), ls ='--', label='fit with pen_dep=' + str(pen_dep[0]*1e9) + 'nm')
                plt.legend(fontsize=14)
                plt.xlabel(r'depth of excited sample [m]', fontsize=16)
                plt.ylabel(r'Absorbed Power density [PW/$m^3$]', fontsize=16)
                if save_fig:
                    assert save_file is not None, 'If you wish to save, please introduce a name for the file with save_file=\'name\''
                    plt.savefig('Results/' + save_file + '.pdf')
                plt.show()

        else:
            dz_sam = self.Sam.get_params_from_blocks('dz')
            sample_depth = np.cumsum(dz_sam)-dz_sam[0]
            plt.xlabel(r'delay [ps]', fontsize=16)
            plt.ylabel(r'sample depth [m]', fontsize=16)
            plt.pcolormesh(self.pulse_time_grid, sample_depth, self.pulse_map.T, cmap='inferno')
            cbar = plt.colorbar()
            cbar.set_label(r'absorbed pump power density [W/m$^3$]', rotation=270, labelpad=15)

            if save_fig:
                assert save_file is not None, 'If you wish to save, please introduce a name for the file with save_file=\'name\''
                plt.savefig('Results/' + save_file + '.png')

            plt.show()

        return

    def show_info(self):
        # This method is called in the main simulation loop to inform the user on the display
        # about the pulse parameters.

        # Input:
        # self (pointer). The pulse object in use.

        # Returns:
        # None. Void function

        print('Absorption profile computed with Abeles\' matrix method.')
        print('F = ', str(self.fluence), ' mJ/cm^2')
        print('F_a_sim =', self.abs_flu, 'mJ/cm^2')
        print('F_r =', self.ref_flu, 'mJ/cm^2')
        print('F_t =', self.trans_flu, 'mJ/cm^2')
        print('F_a = F - F_r - F_t=', self.fluence-self.trans_flu-self.ref_flu, 'mJ/cm^2')
        print('Relative error due to finite layer size: ', np.round(100*(self.abs_flu-(self.fluence-self.trans_flu-self.ref_flu))/self.abs_flu, 2), '%')
        print()
        return
