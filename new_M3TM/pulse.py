import numpy as np
from scipy import constants as sp
from matplotlib import pyplot as plt
from finderb import finderb

class SimPulse:
    # This class lets us define te pulse for excitation of the sample

    def __init__(self, sample, pulse_width, fluence, delay, pulse_dt, method, energy=None, theta=None, phi=None):
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
        # peak_power (float). Peak power per area (intensity) of the pulse in W/m**2.
        # Needed to compute the absorption profile
        # pulse_time_grid, pulse_map (numpy arrays). 1d-arrays of the time-grid on which the pulse is defined
        # and the corresponding 2d-array of excitation power density at all times in all layers

        self.pulse_width = pulse_width
        self.fluence = fluence/10
        self.delay = delay
        self.peak_intensity = self.fluence/np.sqrt(2*np.pi)/self.pulse_width
        self.Sam = sample
        self.pulse_dt = pulse_dt
        self.method = method
        assert self.method == 'LB' or self.method == 'Abele', 'Chose one of the methods \'LB\' (for Lambert-Beer)' \
                                                              ' or \' Abele\' (for computation of Fresnel equations).'
        self.energy = energy
        self.theta = theta
        self.phi = phi
        self.pulse_time_grid, self.pulse_map = self.get_pulse_map()

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
        timestep = self.pulse_dt

        start_pump_time = p_del-10*sigma
        end_pump_time = p_del+10*sigma

        raw_pump_time_grid = np.arange(start_pump_time, end_pump_time, timestep)
        until_pump_start_time = np.arange(0, start_pump_time, timestep)
        pump_time_grid = np.append(until_pump_start_time, raw_pump_time_grid)

        raw_pump_grid = np.exp(-((raw_pump_time_grid-p_del)/sigma)**2/2)
        pump_grid = np.append(np.zeros_like(until_pump_start_time), raw_pump_grid)

        pump_time_grid = np.append(pump_time_grid, end_pump_time+timestep)
        pump_grid = np.append(pump_grid, 0.)

        pump_map = self.depth_profile(pump_grid)

        return pump_time_grid, pump_map

    def depth_profile(self, pump_grid):
        # This method computes the depth dependence of the laser pulse. Either from Lambert-Beer law or from Abele's
        # matrix method.

        # Input:
        # sample (class object). The before constructed sample
        # pump_grid (numpy array). 1d-array of timely pulse shape on the time grid defined in create_pulse_map

        # Returns:
        # 2d-array of the corresponding pump energies on the time grid (first dimension)
        # and for the whole sample (second dimension)

        dz_sam = self.Sam.get_params('dz')
        pendep_sam = self.Sam.get_params('pen_dep')
        mat_blocks = self.Sam.mat_blocks

        max_power = self.peak_intensity
        powers = np.array([])

        if self.method == 'LB':

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
                                    pendep_sam[first_layer:last_layer])
                powers = np.append(powers, max_power/pendep_sam[first_layer:last_layer]
                                   * np.exp(-pen_red))
                max_power = powers[-1]*pendep_sam[last_layer-1]
                first_layer = last_layer
                already_penetrated = 1
            excitation_map = np.multiply(pump_grid[..., np.newaxis], np.array(powers))

            return excitation_map

        elif self.method == 'Abele':

            assert self.Sam.n_comp_arr.any() is not None, 'Please define a refractive index for every constituent'\
                                                               'of the sample within the definition of the sample.'
            assert self.energy is not None and self.theta is not None and self.phi is not None, \
                'For the chosen method, make sure energy, theta and phi are defined.'

            # N is the number of blocks/constituents in the sample, so all in all we have N+2 blocks in the system,
            # considering vacuum before and after the sample:
            N = len(self.Sam.mat_blocks)

            # frequency in 1/s of laser pulse from photon energy:
            wave_length = sp.h*sp.c/sp.physical_constants['electron volt'][0]/self.energy

            # compute the normalized electric field amplitudes from the given angles:
            e_x0 = np.cos(self.phi)*np.cos(self.theta)
            e_y0 = np.sin(self.phi)*np.cos(self.theta)
            e_z0 = np.sin(self.theta)

            # find s and p polarizations from it:
            e_p0 = np.sqrt(e_x0**2+e_z0**2)
            e_s0 = e_y0

            # set up array of refraction indices, first layer and last layer considered vacuum before/after sample:
            n_comp_arr = np.append(np.append(np.ones(1), self.Sam.n_comp_arr), np.ones(1))

            # compute the penetration angle theta in every sample constituent from Snell's law:
            theta_arr = np.empty(N+2, dtype=complex)
            theta_arr[0] = self.theta
            for i, angle in enumerate(theta_arr[1:]):
                angle = np.arcsin(n_comp_arr[i-1]/n_comp_arr[i]*np.sin(theta_arr[i-1]))

            # fresnel equations at N+1 interfaces:
            n_last = n_comp_arr[:-1]
            n_next = n_comp_arr[1:]
            cos_theta_last = np.cos(theta_arr[:-1])
            cos_theta_next = np.cos(theta_arr[1:])


            r_s = np.divide(n_last*cos_theta_last-n_next*cos_theta_next, n_last*cos_theta_last+n_next*cos_theta_next)
            t_s = np.divide(2*n_last*cos_theta_last, n_last*cos_theta_last+n_next*cos_theta_next)
            r_p = np.divide(n_last*cos_theta_next-n_next*cos_theta_last, n_last*cos_theta_next+n_next*cos_theta_last)
            t_p = np.divide(2*n_last*cos_theta_last, n_last*cos_theta_next+n_next*cos_theta_last)

            # we need the thicknesses of blocks and the distance from previous interfaces:
            dzs = self.Sam.get_params('dz')
            penetration_from_interface = np.array([])
            block_thicknesses = np.array([])
            start = 0
            for end in self.Sam.mat_blocks:
                end += start
                penetration_from_interface = np.append(penetration_from_interface, np.cumsum(dzs[start:end])-dzs[start])
                block_thicknesses = np.append(block_thicknesses, sum(dzs[start:end]))
                start = end

            # now the propagation matrices, for N+1 blocks:
            all_C_s_mat = np.empty((N+1, 2, 2), dtype=complex)
            all_C_p_mat = np.empty((N+1, 2, 2), dtype=complex)


            all_C_s_mat[0] = np.array([[1, r_s[0]], [r_s[0], 1]])
            all_C_p_mat[0] = np.array([[1, r_p[0]], [r_p[0], 1]])

            all_phases = 2*np.pi/wave_length*n_comp_arr[1:-1]*cos_theta_last[1:]*block_thicknesses*1j

            for i in range(1, len(all_C_s_mat[1:])):
                all_C_s_mat[i] = 1/t_s[i]*np.array([[np.exp(all_phases[i]), r_s[i]*np.exp(all_phases[i])],
                                         [r_s[i]*np.exp(all_phases[i]), np.exp(all_phases[i])]])
                all_C_p_mat[i] = np.array([[np.exp(all_phases[i]), r_p[i]*np.exp(all_phases[i])],
                                                    [r_p[i]*np.exp(all_phases[i]), np.exp(all_phases[i])]])

            # now D_matrices:
            all_D_s_mat = np.empty((N+2, 2, 2), dtype=complex)
            all_D_p_mat = np.empty((N+2, 2, 2), dtype=complex)

            all_D_s_mat[-1] = np.identity(2)
            all_D_p_mat[-1] = np.identity(2)

            for i in range(len(all_D_p_mat)-2, -1, -1):
                all_D_s_mat[i] = np.matmul(all_D_s_mat[i+1], all_C_s_mat[i])
                all_D_p_mat[i] = np.matmul(all_D_p_mat[i+1], all_C_p_mat[i])

            # total reflection, transmission:

            t_p_tot = np.real(np.divide(np.conj(n_comp_arr[-1]*cos_theta_next[-1]),
                              np.conj(n_comp_arr[0]*cos_theta_last[0])))* np.abs(np.prod(t_p)/all_D_p_mat[-1, 0, 0])**2
            t_s_tot = np.real(np.divide(n_comp_arr[-1]*cos_theta_next[-1], n_comp_arr[0]*cos_theta_last[0]))\
                      * np.abs(np.prod(t_s)/all_D_s_mat[0, 0, 0])**2
            r_s_tot = np.abs(all_D_s_mat[0, 1, 0] / all_D_s_mat[0, 0, 0])**2
            r_p_tot = np.abs(all_D_p_mat[0, 1, 0] / all_D_p_mat[0, 0, 0])**2

            # electric field in all N+2 blocks, for +(-) propagating waves of layer j indexed [j,0(1)]:
            all_E_x_amps = np.empty((N+2, 2), dtype=complex)
            all_E_z_amps = np.empty((N+2, 2), dtype=complex)
            all_E_y_amps = np.empty((N+2, 2), dtype=complex)

            for i in range(N+1):
                tp = np.prod(t_p[:i])
                ts = np.prod(t_s[:i])
                t_p_ex = tp * e_x0
                t_p_ez = tp * e_z0
                t_s_ey = ts * e_y0
                all_E_x_amps[i, 0] = all_D_p_mat[i, 0, 0]/all_D_p_mat[0, 0, 0]*t_p_ex
                all_E_x_amps[i, 1] = all_D_p_mat[i, 1, 0]/all_D_p_mat[0, 0, 0]*t_p_ex
                all_E_z_amps[i, 0] = all_D_p_mat[i, 0, 0]/all_D_p_mat[0, 0, 0]*t_p_ez
                all_E_z_amps[i, 1] = all_D_p_mat[i, 1, 0]/all_D_p_mat[0, 0, 0]*t_p_ez
                all_E_y_amps[i, 0] = all_D_s_mat[i, 0, 0]/all_D_s_mat[0, 0, 0]*t_s_ey
                all_E_y_amps[i, 1] = all_D_s_mat[i, 1, 0]/all_D_s_mat[0, 0, 0]*t_s_ey

            # kz in all blocks of the sample:
            kz_in_sample = 2*np.pi/wave_length*n_comp_arr[1:-1]*np.cos(theta_arr[1:-1])

            # electric field in all layers of sample and proportionality factor for absorptance:
            e_x_in_sample = np.empty((self.Sam.len, 2), dtype=complex)
            e_y_in_sample = np.empty((self.Sam.len, 2), dtype=complex)
            e_z_in_sample = np.empty((self.Sam.len, 2), dtype=complex)
            q_prop = np.empty(self.Sam.len)

            first_layer = 0
            for i, last_layer in enumerate(self.Sam.mat_blocks):
                last_layer += first_layer
                phase = 1j*kz_in_sample[i]*penetration_from_interface[first_layer:last_layer]
                phase = np.array([np.exp(phase), np.exp(-phase)]).T
                e_x_in_sample[first_layer: last_layer] = phase * all_E_x_amps[i+1, :]
                e_y_in_sample[first_layer: last_layer] = phase * all_E_y_amps[i+1, :]
                e_z_in_sample[first_layer: last_layer] = phase * all_E_z_amps[i+1, :]
                q_prop[first_layer: last_layer] = np.real(np.divide(n_comp_arr[i+1]*np.cos(theta_arr[i+1]), np.cos(self.theta)))\
                * 4*np.pi/wave_length*np.imag(n_comp_arr[i+1]*np.cos(theta_arr[i+1]))
                first_layer = last_layer

            # from the E-field we get the normalized intensity:
            p_s_ratio = e_p0**2

            F_z = p_s_ratio*(np.abs(np.sum(e_x_in_sample, axis=-1)**2)+np.abs(np.sum(e_z_in_sample, axis=-1))**2)/e_p0**2 \
                  + (1-p_s_ratio)*(np.abs(np.sum(e_y_in_sample))**2)/e_s0**2

            # and finally the absorbed power densities:
            powers = self.peak_intensity*q_prop*F_z

            excitation_map = np.multiply(pump_grid[..., np.newaxis], powers)

            ### FIX: for phi, theta = [0, pi/2] things crash. add exceptions!!!
            ### ADD: assertion for absorption but no ce defined!!
            ### CHANGE: pen_dep and layer_thickness to sample class, not materials

            return excitation_map

    def visualize(self, key):

        assert key == 't' or key == 'z' or key == 'both', 'Please select one of the three plotting options \'t\' ' \
                                                        '(to see time dependence), \'z\' (to see depth dependence),' \
                                                        '\'both\' (to see the pulse map in time and depth).'
        plt.figure(figsize=(8,6))

        if key == 't':
            norm = np.amax(self.pulse_map)
            j = 0
            max = 0.
            for layer_index in range(self.Sam.len):
                if self.pulse_map[:, layer_index].any() > max:
                    max = np.amax(self.pulse_map[:, layer_index])
                    j = layer_index
            print(norm)
            print(self.pulse_time_grid)
            print(self.pulse_map[:,0])
            plt.plot(self.pulse_time_grid, self.pulse_map[:, j]/norm)
            plt.xlabel(r'delay [ps]', fontsize=16)
            plt.ylabel(r'S(t)/S$_{max}$', fontsize=16)
            plt.show()

        elif key == 'z':
            norm = np.amax(self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :])
            sample_depth = np.cumsum(self.Sam.get_params('dz')) - self.Sam.get_params('dz')[0]
            plt.plot(sample_depth, self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :]/norm)
            plt.xlabel(r'sample depth z [m]', fontsize=16)
            plt.ylabel(r'S(z)/S$_{max}$', fontsize=16)
            plt.show()

        else:
            sample_depth = np.cumsum(self.Sam.get_params('dz'))-self.Sam.get_params('dz')[0]
            plt.xlabel(r'delay [ps]', fontsize=16)
            plt.ylabel(r'sample depth [m]', fontsize=16)
            plt.pcolormesh(self.pulse_time_grid, sample_depth, self.pulse_map.T, cmap='inferno')
            cbar = plt.colorbar()
            cbar.set_label(r'absorbed pump power density [W/m$^3$]', rotation=270, labelpad=15)
            plt.show()

        ### CHANGES: s to ps, m to nm
        ### ADD: Documentation

        return