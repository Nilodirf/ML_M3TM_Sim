import numpy as np
from matplotlib import colors as mplcol
from matplotlib import pyplot as plt
from ..Source.finderb import finderb
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
                thicknesses = line.replace('Layer depth of constituents = ', '').replace('[m]', '')
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
            elif line.startswith('Delay'):
                delay_0 = line.replace('Delay =', '').replace('[s]', '')
                delay_0 = ast.literal_eval(delay_0)
            else:
                if part_of_positions:
                    positions += line.replace('array(', '').replace(')', '').replace('       ', '').replace('\n', ' ')
        positions = positions.replace('[[', '').replace(']]', '')
        positions = positions.split('], [')
        positions = [mat.split(',') for mat in positions]
        positions = np.array([[str(int(pos)+1) for pos in pos_line] for pos_line in positions], dtype=object)
        positions_int = list(np.concatenate(positions))
        positions_in_order = sorted([int(pos) for pos in positions_int])
        label_sorter = [positions_int.index(str(pos)) for pos in positions_in_order]

        positions_te = np.array([[str(int(pos) + 1) for pos in pos_line] for i, pos_line in enumerate(positions) if gamma_els[i] != 0], dtype=object)
        positions_int_te = list(np.concatenate(positions_te))
        positions_in_order_te = sorted([int(pos) for pos in positions_int_te])
        label_sorter_te = [positions_int_te.index(str(pos)) for pos in positions_in_order_te]


        delay -= delay_0

        layer_labels = np.concatenate(np.array([[materials[i] + '_' + position.replace(' ', '')
                                                 for position in positions_line]
                                                 for i, positions_line in enumerate(positions)], dtype=object))[label_sorter]

        layer_labels_te = np.concatenate(np.array([[materials[i] + '_' + position.replace(' ', '')
                                         for position in positions_line]
                                         for i, positions_line in enumerate(positions) if gamma_els[i] != 0], dtype=object))[label_sorter_te]

        if np.array(mu_ats).any() != 0:

            layer_labels_mag = np.concatenate(np.array([[materials[i] + '_' + position.replace(' ', '')
                                              for position in positions_line]
                                              for i, positions_line in enumerate(positions) if mu_ats[i] != 0], dtype=object))
            print(layer_labels_mag)
            layer_thicknesses_mag = np.concatenate(np.array([[thicknesses[i] for _ in positions_line]
                                                             for i, positions_line in enumerate(positions) if
                                                             mu_ats[i] != 0], dtype=object)) * 1e9
        else:
            layer_labels_mag = np.zeros(1)
            layer_thicknesses_mag = np.zeros(1)

        layer_thicknesses = np.concatenate(np.array([[thicknesses[i] for _ in positions_line]
                                           for i, positions_line in enumerate(positions)], dtype=object))*1e9

        layer_thicknesses_te = np.concatenate(np.array([[thicknesses[i] for _ in positions_line]
                                              for i, positions_line in enumerate(positions) if gamma_els[i] != 0], dtype=object))*1e9

        cap_thickness_te = 0.
        for i, l in enumerate(positions):
            if gamma_els[i] == 0:
                cap_thickness_te += thicknesses[i]*len(l)*1e9
            else:
                break

        cap_thickness_mag = 0.
        for i, l in enumerate(positions):
            if mu_ats[i] == 0:
                cap_thickness_mag += thicknesses[i]*len(l)*1e9
            else:
                break

        depth_labels = np.cumsum(layer_thicknesses)-layer_thicknesses[0]

        depth_labels_te = np.cumsum(layer_thicknesses_te) + cap_thickness_te - layer_thicknesses_te[0]

        depth_labels_mag = np.cumsum(layer_thicknesses_mag) + cap_thickness_mag - layer_thicknesses_mag[0]

        return delay.astype(float), tes.astype(float), tps.astype(float), mags.astype(float), layer_labels, layer_labels_te, layer_labels_mag, \
               depth_labels.astype(float), depth_labels_te.astype(float), depth_labels_mag.astype(float)

    def te_tp_plot(self, tp_layers, average, color_scales=['Blues', 'inferno'], min_time=None, max_time=None,
                   save_fig=False, filename=None):

        x = self.delay * 1e12

        if min_time is None:
            min_time = x[0]
        if max_time is None:
            max_time = x[-1]

        first_time_index = finderb(min_time, x)[0]
        last_time_index = finderb(max_time, x)[0]
        x = x[first_time_index: last_time_index]

        tes = self.tes[first_time_index: last_time_index]
        tps = self.tps[first_time_index: last_time_index, tp_layers[0]:tp_layers[1]]

        color = [159 / 255, 77 / 255, 92 / 255]

        if average:
            tes = np.sum(tes, axis=1)/len(tes[0])
            tps = np.sum(tps, axis=1)/len(tps[0])


            plt.figure(figsize=(8, 6))
            plt.plot(x, tes, color=color, linestyle='dashed', label=r'$T_e$', lw=3.0)
            plt.plot(x, tps, color=color, label=r'$T_p$', lw=3.0)

            plt.xlabel(r'delay [ps]', fontsize=16)
            plt.ylabel(r'Temperature [K]', fontsize=16)
            plt.legend(fontsize=14)

            plt.xlim(x[0], x[-1])

        else:

            y_axis = self.depth_labels_te
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
            x_mesh, y_axis_mesh = np.meshgrid(x, y_axis)

            surf_te = ax.plot_surface(x_mesh, y_axis_mesh, tes.T, cmap=color_scales[0],
                                      linewidth=0, antialiased=True, alpha=0.8)
            surf_tp = ax.plot_surface(x_mesh, y_axis_mesh, tps.T, cmap=color_scales[1],
                                      linewidth=0, antialiased=True, alpha=0.8)

            ym, yM = y_axis.min(), y_axis.max()
            tem, teM = tes.min(), tes.max()
            tpm, tpM = tps.min(), tps.max()

            ax.set_zlim(tpm, teM)
            ax.set_ylim(ym, yM)
            ax.set_xlim(x[0], x[-1])
            plt.gca().invert_yaxis()
            cbar_te = plt.colorbar(surf_te, shrink=0.5, aspect=10)
            cbar_tp = plt.colorbar(surf_tp, shrink=0.5, aspect=10)
            cbar_te.set_label(label=r'$T_e$ [K]', fontsize=16)
            cbar_tp.set_label(label=r'$T_p$ [K]', fontsize=16)

            ax.set_xlabel(r'delay [ps]', fontsize=16)
            ax.set_ylabel(r'sample depth [nm]', fontsize=16)

            # add color surface at bottom denoting CGT:
            x_mesh, y_mesh = np.meshgrid((x[0], x[-1]), (y_axis[0], y_axis[-1]))
            z_mesh = np.ones_like(x_mesh)*tpm
            ax.plot_surface(x_mesh, y_mesh, z_mesh, color=color, alpha=0.5)

            # add plots of average temperatures in xz-plane:
            dz = 2e-9
            pen_dep = 30e-9
            exp_decay = np.exp(-np.arange(len(tes.T)) * dz / pen_dep)
            te_av = np.sum(tes * exp_decay, axis=1) / np.sum(exp_decay)
            tp_av = np.sum(tps * exp_decay, axis=1) / np.sum(exp_decay)
            ax.plot(x, yM*np.ones_like(x), te_av, color=color, label=r'$T_e$', ls='dashed', lw=3.0)
            ax.plot(x, yM*np.ones_like(x), tp_av, color=color, label=r'$T_p$', lw=3.0)

            plt.legend(fontsize=14)

            ax.view_init(20, 30)

        if save_fig:
            assert type(filename) == str, 'Denote a filename (path from Results/sim_file) to save the plot.'
            plt.savefig('Results/' + str(self.file) + '/' + str(filename) + '.png')

        plt.show()

    def map_plot(self, key, kind, min_layer=None, max_layer=None, save_fig=False, filename=None, min_time=None,
                 max_time=None, color_scale='inferno', text_color='white', vmin=None, vmax=None, show_title=True):
        # This method creates a color plot with appropriate labeling of one of the simulation output maps.

        # Input:
        # key (string). Chose what you want to plot: `te`, `tp`, `mag` are possible
        # kind (string). Chose 'colormap' or 'surface' to choose the method to show the data.
        # mat_colors (list). Colors that denote the materials at their respective positions
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
        # show_title (boolean). If True, automized title depending on the observable will be shown

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
            y_label_mask = np.array([True if index[0] != y_labels[i+1][0] else False for i, index in enumerate(y_labels[:-1])])
            y_label_mask = np.append(y_label_mask, np.array([True]))
            mat_sep_marks = y_axis[y_label_mask]+2
            # mat_sep_marks -= np.concatenate((np.array([mat_sep_marks[0]]), np.diff(y_axis)))[y_label_mask]
            text_above = [str(y_label)[:y_label.index('_')] for y_label in y_labels[y_label_mask]]
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
            y_label_mask = np.array([True if index[0] != y_labels[i+1][0] else False for i, index in enumerate(y_labels[:-1])])
            y_label_mask = np.append(y_label_mask, np.array([True]))
            mat_sep_marks = y_axis[y_label_mask]+2
            # mat_sep_marks -= np.concatenate((np.array([mat_sep_marks[0]]), np.diff(y_axis)))[y_label_mask] + 2
            text_above = [str(y_label)[:y_label.index('_')] for y_label in y_labels[y_label_mask]]
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

        z_all_layers = z[first_time_index:last_time_index, :]
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
            if show_title:
                plt.title(str(title), fontsize=20)
            cbar = plt.colorbar(label=str(z_label), norm=norm)
            cbar.set_label(str(z_label), rotation=270, labelpad=15)

            for i, mat_sep in enumerate(mat_sep_marks):
                if y_axis[0] < mat_sep < y_axis[-1]:
                    plt.hlines(float(mat_sep), x[0], x[-1], color=text_color)
                plt.text((x[-1]-x[0])*(i+1)/10, float(mat_sep), text_above[i], fontsize=10, color=text_color)

            plt.gca().invert_yaxis()

        if kind == 'surface':

            # x = np.log10(x)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
            x_mesh, y_axis_mesh = np.meshgrid(x, y_axis)
            surf = ax.plot_surface(x_mesh, y_axis_mesh, z.T, cmap=color_scale,
                                   linewidth=0, antialiased=True, alpha=0.8, norm=norm)

            ym, yM = y_axis.min(), y_axis.max()

            ax.set_zlim(vmin, vmax)
            ax.set_ylim(ym, yM)
            ax.set_xlim(x[0], x[-1])

            plt.gca().invert_yaxis()

            cbar = plt.colorbar(surf, shrink=0.5, aspect=10, norm=norm)
            cbar.set_label(label=str(z_label), fontsize=16)
            if show_title:
                plt.title(str(title), fontsize=20)
            ax.set_xlabel(r'delay [ps]', fontsize=14)
            ax.set_ylabel(r'sample depth [nm]', fontsize=14)

            # add surfaces in yz-plane to distinguish sample constituents (we overwrite x_mesh):
            mat_sep_marks = np.append(mat_sep_marks, y_axis[-1])

            colors=['blue', 'red', 'green', 'orange', 'purple']

            if key == 'tp':
                # add surfaces in yz-plane to distinguish sample constituents (we overwrite x_mesh):
                surface_array = np.append(np.zeros(1), mat_sep_marks)
                for i, mark in enumerate(surface_array[:-2]):
                    x_mesh, y_mesh = np.meshgrid((x[0], x[-1]), range(int(mark), int(surface_array[i+1])))
                    z_mesh = np.ones_like(x_mesh)*vmin
                    ax.plot_surface(x_mesh, y_mesh, z_mesh, color=colors[i], alpha=0.5)

                # add lines of the average of tp in each sample constituent, projected onto xz-plane:
                yg = np.ones(x.shape) * yM
                block_separator = np.where(np.array(y_label_mask))[0]
                start_block_at = 0
                for i, pos in enumerate(block_separator[:-1]):
                    end_block_at = pos+start_block_at+1
                    print(start_block_at, end_block_at, colors[i])
                    # if i == 1:
                    #     dz = 2e-9
                    #     pen_dep = 14e-9
                    #     exp_decay = np.exp(-np.arange(block_separator[i+1]-pos) * dz / pen_dep)
                    #     z_block_av = np.sum(z[:, pos:block_separator[i+1]] * exp_decay, axis=1) / np.sum(exp_decay)
                    # else:
                    z_block_av = np.sum(z_all_layers[:, start_block_at:end_block_at], axis=1)/(end_block_at-start_block_at)
                    ax.plot(x, yg, z_block_av, color=colors[i], label=text_above[i], lw=3.0)
                    start_block_at = end_block_at
                # in the substrate, show the average of all layers, regardless of what is shown in the surface plot:
                if len(block_separator) > 1:
                    z_block_av = np.sum(z_all_layers[:, block_separator[-2]:], axis=1) / len(z_all_layers.T[block_separator[-2]:])
                    ax.plot(x, yg, z_block_av, color=colors[-1], label=text_above[-1], lw=3.0)

                # add line at T_C and the text (T_c manual at 65 K):
                # ax.plot(x, yg, np.ones_like(x)*65, color='black', alpha=0.8)
                # ax.text(x[-1], yM, 75, r'$T_C$', color='black', size=14)

                # add grey box at fixed time for zoom effect (for paper):
                # cube_max_x = 6
                # ym = mat_sep_marks[1]
                # yM = mat_sep_marks[2]
                #
                # x_surf_y, x_surf_z = np.meshgrid((ym, yM), (vmin, vmax))
                # x_surf_x = cube_max_x*np.ones_like(x_surf_y)
                # y_surf_x, y_surf_z = np.meshgrid((x[0], cube_max_x), (vmin, vmax))
                # y_surf_y = ym * np.ones_like(y_surf_z)
                # z_surf_x, z_surf_y = np.meshgrid((x[0], cube_max_x), (ym, yM))
                # z_surf_z = vmax*np.ones_like(z_surf_x)

                # ax.plot_surface(x_surf_x, x_surf_y, x_surf_z, color='grey', alpha=0.2)
                # ax.plot_surface(y_surf_x, y_surf_y, y_surf_z, color='grey', alpha=0.4)
                # ax.plot_surface(z_surf_x, z_surf_y, z_surf_z, color='grey', alpha=0.6)

            if key == 'mag' or key == 'te':
                # add surfaces in yz-plane to distinguish sample constituents (we overwrite x_mesh):
                surface_array = np.append(np.zeros(1), mat_sep_marks)
                for i, mark in enumerate(surface_array[:-1]):
                    x_mesh, y_mesh = np.meshgrid((x[0], x[-1]), range(int(mark), int(surface_array[i+1])))
                    z_mesh = np.ones_like(x_mesh)*vmin
                    ax.plot_surface(x_mesh, y_mesh, z_mesh, color=colors[i], alpha=0.5)

                # x_mesh, y_mesh = np.meshgrid((x[0], x[-1]), (y_axis[0], y_axis[-1]))
                # z_mesh = np.ones_like(x_mesh)*vmin
                # ax.plot_surface(x_mesh, y_mesh, z_mesh, color=colors[1], alpha=0.5)

                yg = np.ones(x.shape) * yM
                block_separator = np.where(np.array(y_label_mask))[0]
                start_block_at = 0
                for i, pos in enumerate(block_separator[:-1]):
                    end_block_at = pos+start_block_at+1
                    print(start_block_at, end_block_at, colors[i])
                    # if i == 1:
                    #     dz = 2e-9
                    #     pen_dep = 14e-9
                    #     exp_decay = np.exp(-np.arange(block_separator[i+1]-pos) * dz / pen_dep)
                    #     z_block_av = np.sum(z[:, pos:block_separator[i+1]] * exp_decay, axis=1) / np.sum(exp_decay)
                    # else:
                    z_block_av = np.sum(z_all_layers[:, start_block_at:end_block_at], axis=1)/(end_block_at-start_block_at)
                    ax.plot(x, yg, z_block_av, color=colors[i], label=text_above[i], lw=3.0)
                    start_block_at = end_block_at
                # in the substrate, show the average of all layers, regardless of what is shown in the surface plot:
                if len(block_separator) > 1:
                    z_block_av = np.sum(z_all_layers[:, block_separator[-2]:], axis=1) / len(z_all_layers.T[block_separator[-2]:])
                    ax.plot(x, yg, z_block_av, color=colors[-2], label=text_above[-1], lw=3.0)

                if key == 'mag':
                    yg = yM * np.ones(x.shape)
                    layer_thickness = 2e-9
                    pen_dep = 30e-9
                    exp_decay = np.exp(-np.arange(len(z.T)) * layer_thickness / pen_dep)
                    kerr = np.sum(z*exp_decay, axis=1)/len(z.T)
                    kerr /= kerr[first_time_index]
                    ax.plot(x, yg, kerr, color=colors[1], label=r'CGT', lw=3.0)

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
                    if entry*1e13 % 1 == 0:
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

    def compare_samples(self, key, min_layers, max_layers, colors, labels, save_fig=False, filename=None):

        assert len(min_layers) == len(self.files) and len(max_layers) == len(self.files), 'Introduce as many min ' \
                                                                                          'and max layers as files.'
        assert len(colors) == len(self.files) and len(labels) == len (self.files), 'Introduce as many colors as files.'
        assert key == 'te' or key == 'tp' or key == 'mag', 'Valid keys for plotting subsystems are te, tp and mag.'

        plt.figure(figsize=(8, 6))

        for i, file in enumerate(self.files):
            plot_file = SimPlot(file)
            delays, tes, tps, mags = plot_file.get_data()[:4]
            if key == 'te':
                y = np.sum(tes[:, min_layers[i]: max_layers[i]], axis=1)/(max_layers[i]-min_layers[i])
                y_label = r'$T_e$ [K]'
            elif key == 'tp':
                y = np.sum(tps[:, min_layers[i]: max_layers[i]], axis=1)/(max_layers[i]-min_layers[i])
                y_label = r'$T_p$ [K]'
            elif key == 'mag':
                y = np.sum(mags[:, min_layers[i]: max_layers[i]], axis=1)/(max_layers[i]-min_layers[i])
                y_label = r'magnetization'

            plt.plot(delays*1e12, y, color=colors[i], label=labels[i])

        plt.legend(fontsize=16)
        plt.xlabel(r'delay [ps]', fontsize=20)
        plt.ylabel(y_label, fontsize=20)

        if save_fig == True:
            assert filename is not None and type(filename) == str, 'If you want to save the figure, introuce a filename' \
                                                                   'without file format with filename=...'
            plt.savefig('Results/' + str(filename) + '.pdf')
        plt.show()

        return None