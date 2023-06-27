import numpy as np

class sim_pulse:
    # This class lets us define te pulse for excitation of the sample
    def __init__(self, pulse_width, fluence, delay):
        self.pulse_width = pulse_width
        self.fluence = fluence
        self.delay = delay
        self.peak_power = self.fluence/np.sqrt(2*np.pi)/self.pulse_width*10
        self.pump_time_grid = None
        self.pump_grid = None

    def time_profile(self, end_sim_time):
        # This method creates a time grid and a spatially independent pulse excitation on this time grid.
        # The pulse is confined to nonzero values in the range of [start_pump_time, end_pump_time]
        # to save computation time. After this time, there follows one entry defining zero pump power
        # for all later times.

        # Input:
        # self (object). The pulse defined by the parameters above
        # end_sim_time (float). Maximum simulation time in s

        # Returns:
        # pump_time_grid (numpy array). 1d-array of the time grid on which the pulse is defined
        # pump_grid (numpy array). 1d-array of the corresponding pump energies on the time grid
        pdel = self.delay
        sigma = self.pulse_width
        start_pump_time = pdel-6*sigma
        end_pump_time = pdel+6*sigma

        raw_pump_time_grid = np.arange(start_pump_time, end_pump_time, 1e-16)
        until_pump_start_time = np.arange(0, start_pump_time, 1e-14)
        self.pump_time_grid = np.append(until_pump_start_time,raw_pump_time_grid)

        raw_pump_grid = self.peak_power*np.exp(-((raw_pump_time_grid-pdel)/sigma)**2/2)
        self.pump_grid = np.append(np.zeros_like(until_pump_start_time), raw_pump_grid)

        self.pump_time_grid = np.append(self.pump_time_grid, end_pump_time+5e-15)
        self.pump_grid = np.append(self.pump_grid, 0.)

        return self.pump_time_grid, self.pump_grid

    def depth_profile(self, sample):
        n_sam = sample.get_len()
        dz_sam = sample.get_params('dz')
        pendep_sam = sample.get_params('pen_dep')
        pen_red = np.divide(np.arange(n_sam)*dz_sam, pendep_sam)
        mat_blocks = sample.get_material_changes()

        max_power = self.peak_power
        power_list = []
        first_layer = 0

        for i in range(mat_blocks):
            last_layer = mat_blocks[i]
            power_list.append(max_power/pendep_sam[first_layer:last_layer]*np.exp(pen_red[first_layer:last_layer]))
            max_power = power_list[-1]
            first_layer = last_layer

        excitation_map= np.multiply(self.pump_grid[..., np.newaxis], np.array(power_list))
        return excitation_map


pulse = sim_pulse(pulse_width=20e-15, fluence=1.3, delay=1e-12)
pulse_time_grid, pulse_grid = pulse.time_profile(1e-9)
