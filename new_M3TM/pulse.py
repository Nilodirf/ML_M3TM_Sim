import numpy as np


class SimPulse:
    # This class lets us define te pulse for excitation of the sample

    def __init__(self, sample, pulse_width, fluence, delay, pulse_dt, method, energy=None, theta=None, phi=None):
        # Input:
        # sample (object). Sample in use
        # pulse_width (float). Sigma of gaussian pulse shape in s
        # fluence (float). Fluence of the laser pulse in mJ/cm**2
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
        self.fluence = fluence
        self.delay = delay
        self.peak_power = self.fluence/np.sqrt(2*np.pi)/self.pulse_width*10
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
        timestep = self.pulse_width

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
        # This method computes the depth dependence of the laser pulse in exponential fashion without reflection
        # at interface and multiplies it with the time dependence.

        # Input:
        # sample (class object). The before constructed sample
        # pump_grid (numpy array). 1d-array of timely pulse shape on the time grid defined in create_pulse_map

        # Returns:
        # 2d-array of the corresponding pump energies on the time grid (first dimension)
        # and for the whole sample (second dimension)

        dz_sam = self.Sam.get_params('dz')
        pendep_sam = self.Sam.get_params('pen_dep')
        mat_blocks = self.Sam.mat_blocks

        max_power = self.peak_power
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

            assert self.Sam.get_params('n').any() is not None, 'Please define a refractive index for every constituent'\
                                                               'of the sample within the definition of the materials.'
            assert self.energy is not None and self.theta is not None and self.phi is not None, \
                'For the chosen method, make sure energy, theta and phi are defined.'

            # compute the normalized electric field amplitudes from the given angles:
            e_x0 = np.cos(self.phi)*sin(self.theta)
            e_y0 = np.sin(self.phi)*sin(self.theta)
            e_z0 = np.cos(self.theta)

            # set up array of refraction indices, first layer and last layer considered vacuum before/after sample:
            n_comp_arr = np.append(np.append(np.ones(1), self.Sam.n_comp_arr), np.ones(1))

            # compute the penetration angle theta in every sample constituent from Snell's law:
            theta_arr = np.empty(len(self.Sam.mat_blocks)+2, dtype=complex)
            theta_arr[0]=self.theta
            for i, angle in enumerate(theta_arr[1:]):
                angle = np.arcsin(n_comp_arr[i-1]/n_comp_arr[i]*np.sin(theta_arr[i-1]))

            ### NEXT: FRESNEL EQUATIONS
