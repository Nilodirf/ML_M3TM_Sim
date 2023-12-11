import numpy as np
from matplotlib import colors as mplcol
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from finderb import finderb
import ast


class SimPlot:
    # This class can visualize already saved simulations.
    def __init__(self, file):
        # Input:
        # file (string). The simulation result folder you want to access and plot data from

        # Also returns:
        # delay, tes, tps, mags (numpy arrays). The simulated maps of all three baths (2d except for 1d delay)
        # layer_labels (numpy array). 1d-array of labels for all layers in the materials with material
        # and position relative to the first appearance of the material in the sample
        # layer_labels_te(mag) (numpy arrays). 1d-array of labels for only the layers
        # with free electrons (magnetization dynamics)
        # depth_labels (numpy array). 1d-array of labels for map-plots, containing the depth of the sample in m.
        # depth_labels_te(mag) (numpy arrays). Depth labels for electronic (magnetic) layers in the sample

        self.file = file
        self.delay, self.tes, self.tps, self.mags, \
            self.layer_labels, self.layer_labels_te, self.layer_labels_muat, \
            self.depth_labels, self.depth_labels_te, self.depth_labels_mag = self.get_data()

    def get_data(self):
        # This method loads the data maps from the desired file.

        # Input:
        # self (object). The current plotter object

        # Returns:
        # delay, tes, tps, mags (numpy arrays): The simulated maps of all three baths (2d except for 1d delay)
        # layer_labels (numpy array). 1d-array of labels for all layers in the materials with material
        # and position relative to the first appearance of the material in the sample
        # layer_labels_te(mag) (numpy arrays). 1d-array of labels for only the layers
        # with free electrons (magnetization dynamics)
        # depth_labels (numpy array). 1d-array of labels for map-plots, containing the depth of the sample in m.
        # depth_labels_te(mag) (numpy arrays). Depth labels for electronic (magnetic) layers in the sample

        path = 'Results/' + str(self.file) + '/'
        delay = np.load(path + 'delay.npy')
        tes = np.load(path + 'tes.npy')
        tps = np.load(path + 'tps.npy')
        mags = np.load(path + 'ms.npy')

        with open(path + 'params.dat', 'r') as file:
            content = file.readlines()

        materials = None
        positions = ''
        gamma_els = None
        mu_ats = None
        thicknesses = None
        part_of_positions = False

        for i, line in enumerate(content):
            if line.startswith('Material positions at layer in sample:'):
                positions += line.replace('Material positions at layer in sample: ', '').replace('array(', '').replace(')', '')\
                    .replace('\n', ' ')
                part_of_positions = True
            elif line.startswith('Layer depth'):
                thicknesses = line.replace('Layer depth = ', '').replace('[m]', '')
                thicknesses = ast.literal_eval(thicknesses)
                part_of_positions = False
            elif line.startswith('Materials:'):
                materials = line.replace('Materials: ', '')
                materials = ast.literal_eval(materials)
            elif line.startswith('gamma_el'):
                gamma_els = line.replace('gamma_el =', '').replace('[J/m^3/K^2]', '')
                gamma_els = ast.literal_eval(gamma_els)
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
                                         for i, positions_line in enumerate(positions) if gamma_els[i] != 0]))

        layer_labels_mag = np.concatenate(np.array([[materials[i] + '_' + position.replace(' ', '')
                                          for position in positions_line]
                                          for i, positions_line in enumerate(positions) if mu_ats[i] != 0]))

        layer_thicknesses = np.concatenate(np.array([[thicknesses[i] for _ in positions_line]
                                           for i, positions_line in enumerate(positions)]))*1e9

        layer_thicknesses_te = np.concatenate(np.array([[thicknesses[i] for _ in positions_line]
                                              for i, positions_line in enumerate(positions) if gamma_els[i] != 0]))*1e9

        layer_thicknesses_mag = np.concatenate(np.array([[thicknesses[i] for _ in positions_line]
                                               for i, positions_line in enumerate(positions) if mu_ats[i] != 0]))*1e9

        depth_labels = np.array([np.sum(layer_thicknesses[:i+1]) for i in range(len(layer_thicknesses))])

        depth_labels_te = np.array([np.sum(layer_thicknesses_te[:i+1]) for i in range(len(layer_thicknesses_te))])

        depth_labels_mag = np.array([np.sum(layer_thicknesses_mag[:i + 1]) for i in range(len(layer_thicknesses_mag))])

        return delay, tes, tps, mags, layer_labels, layer_labels_te, layer_labels_mag,\
               depth_labels, depth_labels_te, depth_labels_mag

    def map_plot(self, key, kind, min_layer=None, max_layer=None, save_fig=False, filename=None, min_time=None,
                 max_time=None, color_scale='inferno', text_color='white', vmin=None, vmax=None):
        # This method creates a color plot with appropriate labeling of one of the simulation output maps.

        # Input:
        # key (string). Chose what you want to plot: `te`, `tp`, `mag` are possible
        # kind (string). Chose 'colormap' or 'surface' to choose the method to show the data.
        # save_fig (boolean). If True, the plot will be saved with the according title denoted by
        # filename (String). Default is None
        # min_layer (int). first layer to be plotted. Default is None and then converted to the first layer
        # max_layer (max_layer). last layer to be plotted. Default is None and then converted to the last layer
        # being plotted in the simulation result folder. Default is False
        # min_time (float). The time when the plot should start in ps. Default is None and then converted to the
        # minimal time in self.delay
        # max_time (float). The maximal time that should be plotted in ps. Default is None and then converted
        # to the maximum time in self.delay
        # color_scale (string). One of the python in-built color-scales to show the data. Default is 'inferno'
        # text_color (string). One of the standard colors in which the materials and horizontal lines denoting material
        # separation will be written. Default is 'white'
        # vmin (float). The minimum z-value to which the color map will be scaled. If not specified, default is None
        # and will be converted to the minimum temperature (or magnetization) in the selected dataset to be plotted.
        # vmax (float). The maximum z-value to which the color map will be scaled. If not specified, default is None
        # and will be converted to the maximum temperature (or magnetization) in the selected dataset to be plotted.


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
            (M0, N0) = z.shape
            if min_layer is None:
                min_layer = 0
            if max_layer is None:
                max_layer = N0
            title = 'Electron Temperature Map'
            y_axis = self.depth_labels_te[min_layer: max_layer]
            y_labels = self.layer_labels_te[min_layer: max_layer]
            y_label_mask = np.array([True if index.endswith('_1') else False for index in y_labels])
            mat_sep_marks = y_axis[y_label_mask]
            mat_sep_marks -= np.concatenate((np.array([mat_sep_marks[0]]), np.diff(y_axis)))[y_label_mask]
            text_above = [str(y_label).replace('_1', '') for y_label in y_labels[y_label_mask]]
            z_label = r'$T_e$ [K]'
        elif key == 'tp':
            z = self.tps
            (M0, N0) = z.shape
            if min_layer is None:
                min_layer = 0
            if max_layer is None:
                max_layer = N0
            title = 'Phonon Temperature Map'
            y_axis = self.depth_labels[min_layer: max_layer]
            y_labels = self.layer_labels[min_layer:max_layer]
            y_label_mask = np.array([True if index.endswith('_1') else False for index in y_labels])
            mat_sep_marks = y_axis[y_label_mask]
            mat_sep_marks -= np.concatenate((np.array([mat_sep_marks[0]]), np.diff(y_axis)))[y_label_mask]
            text_above = [str(y_label).replace('_1', '') for y_label in y_labels[y_label_mask]]
            z_label = r'$T_p$ [K]'
        elif key == 'mag':
            z = self.mags
            (M0, N0) = z.shape
            if min_layer is None:
                min_layer = 0
            if max_layer is None:
                max_layer = N0
            title = 'Magnetization Map'
            y_axis = self.depth_labels_mag[min_layer: max_layer]
            y_labels = self.layer_labels_muat[min_layer: max_layer]
            y_label_mask = np.array([True if '1' in index else False for index in y_labels])
            mat_sep_marks = y_axis[y_label_mask]
            mat_sep_marks -= np.concatenate((np.array([mat_sep_marks[0]]), np.diff(y_axis)))[y_label_mask]
            text_above = [str(y_label).replace('_1', '') for y_label in y_labels[y_label_mask]]
            z_label = 'Magnetization'
        else:
            print('In SimPlot.map_plot(): Please enter a valid key: You can either plot ´te´, ´tp´ or ´mag´.')
            return

        z = z[first_time_index:last_time_index, min_layer: max_layer]

        if vmin is None:
            vmin = np.amin(z)
        if vmax is None:
            vmax = np.amax(z)

        norm = mplcol.Normalize(vmin=vmin, vmax=vmax)

        if kind == 'colormap':

            plt.figure(figsize=(8, 6))

            plt.pcolormesh(x, y_axis, z.T, cmap=color_scale, norm=norm)

            plt.xlabel(r'time [ps]', fontsize=16)
            plt.ylabel(r'sample depth [nm]', fontsize=16)
            plt.title(str(title), fontsize=20)
            cbar = plt.colorbar(label=str(z_label), norm=norm)
            cbar.set_label(str(z_label), rotation=270, labelpad=15)

            for i, mat_sep in enumerate(mat_sep_marks):
                if y_axis[0] < mat_sep < y_axis[-1]:
                    plt.hlines(float(mat_sep), x[0], x[-1], color=text_color)
                plt.text((x[-1]-x[0])*0.5/10, float(mat_sep) + 13, text_above[i], fontsize=14, color=text_color)

            plt.gca().invert_yaxis()


        if kind == 'surface':

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
            x_mesh, y_axis_mesh = np.meshgrid(x, y_axis)
            surf = ax.plot_surface(x_mesh, y_axis_mesh, z.T, cmap=color_scale,
                                   linewidth=0, antialiased=True)

            ax.set_zlim(vmin, vmax)
            ax.set_ylim(y_axis[0], y_axis[-1])
            plt.gca().invert_yaxis()
            plt.colorbar(surf, label=str(z_label), shrink=0.5, aspect=5, norm=norm)

            plt.title(str(title), fontsize=20)
            ax.set_xlabel(r'delay [ps]', fontsize=14)
            ax.set_ylabel(r'sample depth [nm]', fontsize=14)

            # add surfaces in yz-plane to distinguish sample constituents (we overwrite x_mesh):
            colors = [[77 / 255, 77 / 255, 159 / 255], [159 / 255, 77 / 255, 92 / 255], [89 / 255, 159 / 255, 77 / 255]]
            mat_sep_marks = np.append(mat_sep_marks, y_axis[-1])
            for i, mark in enumerate(mat_sep_marks[1:]):
                x_mesh, y_mesh = np.meshgrid(range(int(x[-1])+3), range(int(mat_sep_marks[i]), int(mark)))
                z_mesh = np.ones_like(x_mesh)*np.amin(z)
                ax.plot_surface(x_mesh, y_mesh, z_mesh, color=colors[i], alpha=0.5)

            # add lines of the average of tp in each sample constituent:
            if key == 'tp':
                ym, yM = y_axis.min(), y_axis.max()
                yg = yM * np.ones(x.shape)
                block_separator = np.where(np.array(y_label_mask))[0]
                for i, pos in enumerate(block_separator[:-1]):
                    z_block_av = np.sum(z[:, pos:pos+1], axis=1)/(pos+1-pos)
                    ax.plot(x, yg, z_block_av, color=colors[i], label=text_above[i])
                z_block_av = np.sum(z[:, block_separator[-1]:], axis=1) / len(y_label_mask[block_separator[-1]:])
                ax.plot(x, yg, z_block_av, color=colors[-1], label=text_above[-1])

                # add line at T_C and the text (T_c manual at 65 K):
                ax.plot(x, yg, np.ones_like(x)*65, color='grey', alpha=0.5)
                ax.text(x[-1], yM, 75, r'$T_C$', color='grey', size=14)

            ax.view_init(20, 30)

            plt.legend()

        if save_fig:
            assert type(filename) == str, 'Denote a filename (path from Results/sim_file) to save the plot.'
            plt.savefig('Results/' + str(self.file) + '/' + str(filename) + '.png')

        plt.show()


        return

    def line_plot(self, key, average=False, min_layer=None, max_layer=None,
                  save_fig=False, min_time=None, max_time=None, norm=False):
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
        # norm (boolean). If key == 'mag', average is False and norm is True, the magnetization dynamics of all layers
        # will be normalized in a way that the layer with the largest demagnetization will demagnetize from 1 to 0
        # in the selected time and layer range.

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

        else:

            if key == 'mag' and norm:
                y_label = 'Normalized magnetization'
                norm_factor = 0
                for i in range(min_layer, max_layer+1):
                    max_difference_in_layer = np.abs(np.amin(y[first_time_index:last_time_index, i]
                                                             - y[first_time_index, i]))
                    if max_difference_in_layer > norm_factor:
                        norm_factor = max_difference_in_layer
                if norm_factor > 0:
                    y = 1 + (y-y[first_time_index])/norm_factor

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

    def convert_to_dat(self):
        # This method converts the simulation results that are initiated with this class as arrays from .npy files
        # to .dat files and stores them with the same name in the same directory.

        # Input:
        # self. The plot object in use

        # Returns:
        # None. The method is void, only the .dat files are created and written.

        # Define path to save files:
        path = 'Results/' + str(self.file) + '/'

        # Define files:
        delay_dat_file = open(path + 'delays.dat', 'w+')
        te_dat_file = open(path + 'tes.dat', 'w+')
        tp_dat_file = open(path + 'tps.dat', 'w+')
        mag_dat_file = open(path + 'mag.dat', 'w+')

        # Filter the results to write in .dat to save memory (every 10 fs):
        time_in_ps = self.delay*1e12
        time_increment = 1e-2
        write_times = np.arange(0,time_in_ps[-1], time_increment)
        write_time_indices_unfiltered = finderb(write_times, time_in_ps)
        times_to_write_unfiltered = np.round(self.delay, 14)[write_time_indices_unfiltered]
        times_to_write = []
        write_time_indices = []

        # If time increments in the data are larger than 10 fs, filter out datapoints recorded several times:
        for i, entry in enumerate(times_to_write_unfiltered):
            if entry not in times_to_write:
                if entry <= 10e-12:
                    times_to_write.append(entry)
                    write_time_indices.append(write_time_indices_unfiltered[i])
                else:
                    if entry*1e13 % 1 ==0:
                        times_to_write.append(entry)
                        write_time_indices.append(write_time_indices_unfiltered[i])

        # Write the according data in the files:
        # After every time step, defined by the filter write_mask, a linebreak is forced in the .dat files.
        for i in range(len(times_to_write)):
            delay_dat_file.write(str(times_to_write[i]) + '\n')
            te_dat_file.write(str([np.round(te_loc, 4) for te_loc in self.tes[write_time_indices][i]]).replace('[', '')
                              .replace(']', '').replace(',', '\t') + '\n')
            tp_dat_file.write(str([np.round(tp_loc, 4) for tp_loc in self.tps[write_time_indices][i]]).replace('[', '')
                              .replace(']', '').replace(',', '\t') + '\n')
            mag_dat_file.write(str([np.round(mag_loc, 4) for mag_loc in self.mags[write_time_indices][i]]).replace('[', '')
                               .replace(']', '').replace(',', '\t') + '\n')

        # Close the files:
        delay_dat_file.close()
        te_dat_file.close()
        tp_dat_file.close()
        mag_dat_file.close()

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
        tes = np.load(path + 'tes.npy')
        tps = np.load(path + 'tps.npy')

        return delay, mags, tes, tps

    def kerr_plot(self, pen_dep, layer_thickness, min_time, max_time, save_fig=False, norm=False, filename=None):
        # This method plots a Kerr-signal, exponentially weighting the magnetization of each layer in the dataset,
        # normalizing the data and plotting several data on top of each other.

        # Input:
        # self (object). The current plotter object
        # pen_dep (float). Penetration depth of the probe pulse in m
        # layer thickness. Thickness of the magnetic layers. (This could be automized in a more complicated fashion,
        # might do this in the future)
        # min_time (float). The time when the plot should start in ps
        # max_time (float). The maximal time that should be plotted in ps
        # save_fig (boolean). If True, the plot will be saved with the according title denoting the files that are
        # being plotted in the Results folder. Default is False
        # filename (str). Name the file if save_fig=True. It will be saved in .pdf format automatically

        # Returns:
        # None. After creating the plot and possibly saving it, the functions returns nothing

        plt.figure(figsize=(8, 6))

        for file in self.files:
            delay, mags = SimComparePlot.get_data(file)[0:2]

            delay = delay*1e12

            first_time_index = finderb(min_time, delay)[0]
            last_time_index = finderb(max_time, delay)[0]
            delay = delay[first_time_index: last_time_index]

            zero_time = finderb(0., delay)[0]

            exp_decay = np.exp(-np.arange(len(mags.T))*layer_thickness/pen_dep)
            kerr_signal = np.sum(np.multiply(mags, exp_decay[np.newaxis, ...]), axis=1)

            kerr_in_time = kerr_signal[first_time_index:last_time_index]

            if norm:
                norm_kerr_signal = (kerr_in_time-kerr_signal[zero_time])\
                                    / np.abs(np.amin(kerr_signal-kerr_signal[zero_time]))
            else:
                norm_kerr_signal = kerr_in_time/kerr_in_time[zero_time]

            plt.plot(delay, norm_kerr_signal, label=str(file))

        plt.xlabel(r'delay [ps]', fontsize=16)
        plt.ylabel(r'Norm. Kerr signal', fontsize=16)
        plt.title(r'MOKE Simulation', fontsize=20)
        plt.legend(fontsize=14)

        if save_fig:
            assert type(filename) == str, ('Select a filename for the save_file if you wish'
                                           ' saving the plot on the hard_drive.')
            plt.savefig('Results/' + filename + '.pdf')

        plt.show()

        return

    def parameter_2d_scan(self):

        plt.figure(figsize=(8, 6))

        for file in self.files:
            delay, mags = SimComparePlot.get_data(file)[0:2]

            delay = delay * 1e12

            zero_time = finderb(0., delay)[0]




        return None
