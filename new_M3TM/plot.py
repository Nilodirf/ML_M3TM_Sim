import numpy as np
from matplotlib import pyplot as plt
from finderb import finderb
import ast


class SimPlot:
    # This class can visualize already saved simulations.
    def __init__(self, file):
        # Input:
        # file (string). The simulation result folder you want to access and plot data from

        # Also returns:
        # delay, tes, tps, mags (numpy arrays). The simulated maps of all three baths (2d except for 1d delay)
        # layer_labels (list). List of labels for all layers in the materials with material and position relative
        # to the first appearance of the material in the sample

        self.file = file
        self.delay, self.tes, self.tps, self.mags,\
            self.layer_labels, self.layer_labels_te, self.layer_labels_muat = self.get_data()

    def get_data(self):
        # This method loads the data maps from the desired file.

        # Input:
        # self (object). The current plotter object

        # Returns:
        # delay, tes, tps, mags (numpy arrays): The simulated maps of all three baths (2d except for 1d delay)

        path = 'Results/' + str(self.file) + '/'
        delay = np.load(path + 'delay.npy')
        tes = np.load(path + 'tes.npy')
        tps = np.load(path + 'tps.npy')
        mags = np.load(path + 'ms.npy')

        with open(path + 'params.dat', 'r') as file:
            content = file.readlines()

        materials = None
        positions = ''
        kappa_els = None
        mu_ats = None
        part_of_positions = False

        for i, line in enumerate(content):
            if line.startswith('Material positions in order:'):
                positions += line.replace('Material positions in order: ', '').replace('array(', '').replace(')', '')\
                    .replace('\n', ' ')
                part_of_positions = True
            elif line.startswith('Layer depth'):
                part_of_positions = False
            elif line.startswith('Materials:'):
                materials = line.replace('Materials: ', '')
                materials = ast.literal_eval(materials)
            elif line.startswith('kappa_el')  :
                kappa_els = line.replace('kappa_el =', '').replace('[W/mK]', '')
                kappa_els = ast.literal_eval(kappa_els)
            elif line.startswith('mu_at'):
                mu_ats = line.replace('mu_at = ', '').replace('[mu_Bohr]', '')
                mu_ats = ast.literal_eval(mu_ats)
            else:
                if part_of_positions:
                    positions += line.replace('array(', '').replace(')', '').replace('       ', '').replace('\n', ' ')
        positions = positions.replace('[[', '').replace(']]', '')
        positions = positions.split('], [')
        positions = [mat.split(',') for mat in positions]
        positions = [[str(int(pos)-int(pos_line[0]) + 1) for pos in pos_line] for pos_line in positions]

        layer_labels = np.concatenate(np.array([[materials[i] + '_' + position.replace(' ', '')
                                                 for position in positions_line]
                                                 for i, positions_line in enumerate(positions)]))

        layer_labels_te = np.concatenate(np.array([[materials[i] + '_' + position.replace(' ', '')
                                         for position in positions_line]
                                         for i, positions_line in enumerate(positions) if mu_ats[i] != 0]))

        layer_labels_mag = np.concatenate(np.array([[materials[i] + '_' + position.replace(' ', '')
                                          for position in positions_line]
                                          for i, positions_line in enumerate(positions) if kappa_els[i] != 0]))

        return delay, tes, tps, mags, layer_labels, layer_labels_te, layer_labels_mag

    def map_plot(self, key, min_layer=None, max_layer=None, save_fig=False, min_time=None, max_time=None):
        # This method creates a color plot with appropriate labeling of one of the simulation output maps.

        # Input:
        # key (string). Chose what you want to plot: `te`, `tp`, `mag` are possible
        # save_fig (boolean). If True, the plot will be saved with the according title denoting the map that is
        # min_layer (int). first layer to be plotted. Default is None and then converted to the first layer
        # max_layer (max_layer). last layer to be plotted. Default is None and then converted to the last layer
        # being plotted in the simulation result folder. Default is False
        # min_time (float). The time when the plot should start in ps. Default is None and then converted to the
        # minimal time in self.delay
        # max_time (float). The maximal time that should be plotted in ps. Default is None and then converted
        # to the maximum time in self.delay

        # Returns:
        # None. After creating the plot and possibly saving it, the functions returns nothing

        x = self.delay * 1e12

        if min_time is None:
            min_time = x[0]
        if max_time is None:
            max_time = x[-1]

        first_time_index = finderb(min_time, x)[0]
        last_time_index = finderb(max_time, x)[0]
        x = x[first_time_index: last_time_index]

        if key == 'te':
            z = self.tes
            title = 'Electron Temperature Map'
            z_label = 'T_e [K]'
        elif key == 'tp':
            z = self.tps
            title = 'Phonon Temperature Map'
            z_label = 'T_p [K]'
        elif key == 'mag':
            z = self.mags
            title = 'Magnetization Map'
            z_label = 'Magnetization'
        else:
            print('In SimPlot.map_plot(): Please enter a valid key: You can either plot ´te´, ´tp´ or ´mag´.')
            return

        plt.figure(figsize=(8, 6))

        (M0, N0) = z.shape
        if min_layer is None:
            min_layer = 0
        if max_layer is None:
            max_layer = N0

        plt.pcolormesh(x, np.arange(min_layer, max_layer), z[first_time_index:last_time_index, min_layer: max_layer].T,
                       cmap='jet')
        plt.xlabel(r'time [ps]', fontsize=16)
        plt.ylabel(r'layer', fontsize=16)
        plt.title(str(title), fontsize=20)
        cbar = plt.colorbar(label=str(z_label))
        cbar.set_label(str(z_label), rotation=270, labelpad=15)

        if save_fig:
            plt.savefig('Results/' + str(self.file) + '/' + str(title) + '.pdf')

        plt.show()

        return

    def line_plot(self, key, average=False, min_layer=None, max_layer=None,
                  save_fig=False, min_time=None, max_time=None):
        # This method produces line plots of the dynamics of a desired subsystem and for a number of desired layers.

        # Input:
        # key (string). Chose what you want to plot: `te`, `tp`, `mag` are possible
        # average (boolean). If true, the mentioned data is averaged over the selected layers. Default is False
        # min_layer (int). first layer to be plotted. Default is None and then converted to the first layer
        # max_layer (max_layer). last layer to be plotted. Default is None and then converted to the last layer
        # save_fig (boolean). If True, the plot will be saved with the according title denoting the map that is
        # being plotted in the simulation result folder.Default is False
        # min_time (float). The time when the plot should start in ps. Default is None and then converted to the
        # minimal time in self.delay
        # max_time (float). The maximal time that should be plotted in ps. Default is None and then converted
        # to the maximum time in self.delay

        # Returns:
        # None. After creating the plot and possibly saving it, the functions returns nothing

        x = self.delay * 1e12

        if min_time is None:
            min_time = x[0]
        if max_time is None:
            max_time = x[-1]

        first_time_index = finderb(min_time, x)[0]
        last_time_index = finderb(max_time, x)[0]
        x = x[first_time_index: last_time_index]

        if key == 'te':
            y = self.tes
            title = 'Electron Temperature Dynamics'
            y_label = 'T_e [K]'
            line_labels = self.layer_labels_te[min_layer: max_layer]
        elif key == 'tp':
            y = self.tps
            title = 'Phonon Temperature Dynamics'
            y_label = 'T_p [K]'
            line_labels = self.layer_labels[min_layer: max_layer]
        elif key == 'mag':
            y = self.mags
            title = 'Magnetization Dynamics'
            y_label = 'Magnetization'
            line_labels = self.layer_labels_muat[min_layer: max_layer]
        else:
            print('In SimPlot.map_plot(): Please enter a valid key: You can either plot ´te´, ´tp´ or ´mag´.')
            return

        (M0, N0) = y.shape
        if min_layer is None:
            min_layer = 0
        if max_layer is None:
            max_layer = N0

        plt.figure(figsize=(8, 6))

        if average:
            plt.plot(x, np.sum(y[first_time_index:last_time_index, min_layer:max_layer], axis=1)/(max_layer-min_layer))
            layer_labels = []
        else:
            for i in range(min_layer, max_layer):
                plt.plot(x, y[first_time_index:last_time_index, i])
            plt.legend(line_labels, fontsize=14)
        plt.xlabel(r'delay [ps]', fontsize=16)
        plt.ylabel(str(y_label), fontsize=16)
        plt.title(str(title), fontsize=20)

        if save_fig:
            plt.savefig('Results/' + str(self.file) + '/' + str(title) + '.pdf')

        plt.show()

        return


class SimComparePlot:
    # This class enables comparison of different data sets.

    def __init__(self, files):
        self.files = files

    @staticmethod
    def get_data(file):
        # This method loads the data maps from the desired file.

        # Input:
        # self (object). The current plotter object

        # Returns:
        # delay, tes, tps, mags (numpy arrays): The simulated maps of all three baths (2d except for 1d delay)

        path = 'Results/' + str(file) + '/'
        delay = np.load(path + 'delay.npy')
        mags = np.load(path + 'ms.npy')

        return delay, mags

    def kerr_plot(self, pen_dep, layer_thickness, min_time, max_time, save_fig=False):
        # This method plots a Kerr-signal, exponentially weighting the magnetization of each layer in the dataset,
        # normalizing the data and plotting several data on top of eachother.

        # Input:
        # self (object). The current plotter object
        # pen_dep (float). Penetration depth of the probe pulse in m
        # layer thickness. Thickness of the magnetic layers. (This could be automized in a more complicated fashion,
        # might do this in the future)
        # min_time (float). The time when the plot should start in ps
        # max_time (float). The maximal time that should be plotted in ps
        # save_fig (boolean). If True, the plot will be saved with the according title denoting the files that are
        # being plotted in the Results folder. Default is False

        # Returns:
        # None. After creating the plot and possibly saving it, the functions returns nothing

        plt.figure(figsize=(8, 6))

        for file in self.files:
            delay, mags = SimComparePlot.get_data(file)

            delay = delay*1e12

            first_time_index = finderb(min_time, delay)[0]
            last_time_index = finderb(max_time, delay)[0]
            delay = delay[first_time_index: last_time_index]

            zero_time = finderb(0., delay)[0]

            exp_decay = np.exp(-np.arange(len(mags.T))*layer_thickness/pen_dep)
            kerr_signal = np.sum(np.multiply(mags, exp_decay[np.newaxis, ...]), axis=1)

            kerr_in_time = kerr_signal[first_time_index:last_time_index]

            norm_kerr_signal = (kerr_in_time-kerr_signal[zero_time])\
                                / np.abs(np.amin(kerr_signal-kerr_signal[zero_time]))

            plt.plot(delay, norm_kerr_signal, label=str(file))

        plt.xlabel(r'delay [ps]', fontsize=16)
        plt.ylabel(r'Norm. Kerr signal', fontsize=16)
        plt.title(r'MOKE Simulation', fontsize=20)
        plt.legend(fontsize=14)

        if save_fig:
            plt.savefig('Results/' + str(self.files) + '_Kerr/' + '.pdf')

        plt.show()

        return





