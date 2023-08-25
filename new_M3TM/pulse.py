import numpy as np


class SimPulse:
    # This class lets us define te pulse for excitation of the sample

    def __init__(self, sample, pulse_width, fluence, delay, pulse_dt):
        # Input:
        # sample (object). Sample in use
        # pulse_width (float). Sigma of gaussian pulse shape in s
        # fluence (float). Fluence of the laser pulse in mJ/cm**2
        # delay (float). time-delay of the pulse peak after simulation start in s (Let the magnetization relax
        # to equilibrium before injecting the pulse

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
