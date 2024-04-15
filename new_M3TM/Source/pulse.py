import numpy as np
from matplotlib import pyplot as plt

from ..Source.finderb import finderb


class SimPulse:
    # This class lets us define te pulse for excitation of the sample

    def __init__(self, sample, pulse_width, fluence, delay):
        # Input:
        # sample (object). Sample in use
        # pulse_width (float). Sigma of gaussian pulse shape in s
        # fluence (float). Fluence of the laser pulse in mJ/cm**2
        # delay (float). time-delay of the pulse peak after simulation start in s (Let the magnetization relax
        # to equilibrium before injecting the pulse

        # Also returns:
        # peak_power (float). Peak power per area of the pulse in W/m**2. Needed to compute the absorption profile.#
        # pulse_time_grid, pulse_map (numpy arrays). 1d-arrays of the time-grid on which the pulse is defined
        # and the corresponding 2d-array of excitation power density at all times in all layers.

        self.pulse_width = pulse_width
        self.fluence = fluence
        self.delay = delay
        self.peak_power = self.fluence/np.sqrt(2*np.pi)/self.pulse_width*10
        self.Sam = sample
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
        start_pump_time = p_del-10*sigma
        end_pump_time = p_del+10*sigma

        raw_pump_time_grid = np.arange(start_pump_time, end_pump_time, 1e-16)
        until_pump_start_time = np.arange(0, start_pump_time, 1e-16)
        pump_time_grid = np.append(until_pump_start_time, raw_pump_time_grid)

        raw_pump_grid = np.exp(-((raw_pump_time_grid-p_del)/sigma)**2/2)
        pump_grid = np.append(np.zeros_like(until_pump_start_time), raw_pump_grid)

        pump_time_grid = np.append(pump_time_grid, end_pump_time+1e-15)
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

        for i in range(len(mat_blocks)):
            last_layer += mat_blocks[i]
            if pendep_sam[first_layer] == 1:
                powers = np.append(powers, np.zeros(mat_blocks[i]))
                first_layer = last_layer
                continue
            pen_red = np.divide(np.arange(mat_blocks[i])*dz_sam[first_layer:last_layer],
                                pendep_sam[first_layer:last_layer])
            powers = np.append(powers, max_power/pendep_sam[first_layer:last_layer]
                               * np.exp(-pen_red))
            max_power = powers[-1]*pendep_sam[last_layer-1]
            first_layer = last_layer
        excitation_map = np.multiply(pump_grid[..., np.newaxis], np.array(powers))

        return excitation_map

    def visualize(self, axis, fit=None, save_fig=False, save_file=None):
        # This method plots the spatial/temporal/both dependencies of the pump pulse.

        # Input:
        # self (object). The pulse object in use
        # axis (String). Chose whether to plot the temporal profile ('t'), the spatial profile of absorption ('z')
        # or both ('tz').

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
            norm = np.amax(self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :])
            sample_depth = np.cumsum(self.Sam.get_params('dz')) - self.Sam.get_params('dz')[0]
            plt.plot(sample_depth*1e9, self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :]/norm)
            plt.xlabel(r'sample depth z [nm]', fontsize=16)
            plt.ylabel(r'S(z)/S$_{max}$', fontsize=16)
            if save_fig:
                assert save_file is not None, 'If you wish to save, please introduce a name for the file with save_file=\'name\''
                plt.savefig('Results/' + save_file + '.pdf')
            plt.show()

            if fit is not None:
                if fit == 'exp':
                    def fit_func(depth, pen_dep):
                        return np.exp(-depth/pen_dep)
                elif fit == 'lin':
                    def fit_func(depth, slope):
                        return 1-(slope*depth)
                excited_depth = sample_depth[self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :] != 0]
                excited_depth -= excited_depth[0]
                to_fit = self.pulse_map[finderb(self.delay, self.pulse_time_grid)[0], :] / norm
                to_fit = to_fit[to_fit != 0]
                p0 = 30e-9
                pen_dep, cv = scipy.optimize.curve_fit(fit_func, excited_depth, to_fit, p0)
                plt.plot(excited_depth, to_fit, ls='dotted', lw=3, label='Abeles\' method')
                plt.plot(excited_depth, fit_func(excited_depth, pen_dep[0]), ls ='--', label='fit with pen_dep=' + str(pen_dep[0]*1e9) + 'nm')
                plt.legend(fontsize=14)
                plt.xlabel(r'depth of excited sample [m]', fontsize=16)
                plt.ylabel(r'Normalized power', fontsize=16)
                if save_fig:
                    assert save_file is not None, 'If you wish to save, please introduce a name for the file with save_file=\'name\''
                    plt.savefig('Results/' + save_file + '.pdf')
                plt.show()

        else:
            sample_depth = np.cumsum(self.Sam.get_params('dz'))-self.Sam.get_params('dz')[0]
            plt.xlabel(r'delay [ps]', fontsize=16)
            plt.ylabel(r'sample depth [m]', fontsize=16)
            plt.pcolormesh(self.pulse_time_grid, sample_depth, self.pulse_map.T, cmap='inferno')
            cbar = plt.colorbar()
            cbar.set_label(r'absorbed pump power density [W/m$^3$]', rotation=270, labelpad=15)

            if save_fig:
                assert save_file is not None, 'If you wish to save, please introduce a name for the file with save_file=\'name\''
                plt.savefig('Results/' + save_file + '.png')

            plt.show()

        ### CHANGES: s to ps, m to nm, full power

        return
