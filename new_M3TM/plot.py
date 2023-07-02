import numpy as np
from matplotlib import pyplot as plt


class SimPlot:
    # This class can visualize already saved simulations.
    def __init__(self, file):
        # Input:
        # file (string). The simulation result folder you want to access and plot data from

        # Also returns:
        # delay, tes, tps, mags (numpy arrays): The simulated maps of all three baths (2d except for 1d delay)

        self.file = file
        self.delay, self.tes, self.tps, self.mags = self.get_data()

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

        return delay, tes, tps, mags

    def map_plot(self, key, save_fig):
        # This method creates a color plot with appropriate labeling of one of the simulation output maps.

        # Input:
        # key (string). Chose what you want to plot: `te`, `tp`, `mag` are possible
        # save_fig (boolean). If True, the plot will be saved with the according title denoting the map that is
        # being plotted in the simulation result folder.

        # Returns:
        # None. After creating the plot and possibly saving it, the functions returns nothing.

        x = self.delay * 1e12
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

        (M, N) = z.shape

        plt.imshow(z.T, aspect='auto', origin='lower', extent=[x[0], x[-1], 0, N-1], cmap='jet')
        plt.xlabel(r'time [ps]', fontsize=16)
        plt.ylabel(r'layer', fontsize=16)
        plt.title(str(title), fontsize=20)
        cbar = plt.colorbar(label=str(z_label))
        cbar.set_label(str(z_label), rotation=270, labelpad=15)

        if save_fig:
            plt.savefig('Results/' + str(self.file) + '/' + str(title) + '.pdf')

        plt.show()

        return

    def line_plot(self, key, min_layer, max_layer, save_fig):
        # This method produces line plots of the dynamics of a desired subsystem and for a number of desired layers.

        # Input:
        # key (string). Chose what you want to plot: `te`, `tp`, `mag` are possible
        # min_layer (int). first layer to be plotted
        # max_layer (max_layer). last layer to be plotted
        # save_fig (boolean). If True, the plot will be saved with the according title denoting the map that is
        # being plotted in the simulation result folder.

        # Returns:
        # None. After creating the plot and possibly saving it, the functions returns nothing.

        x = self.delay * 1e12
        if key == 'te':
            y = self.tes
            title = 'Electron Temperature Dynamics'
            y_label = 'T_e [K]'
        elif key == 'tp':
            y = self.tps
            title = 'Phonon Temperature Dynamics'
            y_label = 'T_p [K]'
        elif key == 'mag':
            y = self.mags
            title = 'Magnetization Dynamics'
            y_label = 'Magnetization'
        else:
            print('In SimPlot.map_plot(): Please enter a valid key: You can either plot ´te´, ´tp´ or ´mag´.')
            return

        plt.figure(figsize=(8, 6))

        for i in range(min_layer, max_layer):
            plt.plot(x, y[:, i], label='layer '+str(i))
        plt.xlabel(r'delay [ps]', fontsize=16)
        plt.ylabel(str(y_label), fontsize=16)
        plt.legend(fontsize=14)
        plt.title(str(title), fontsize=20)

        if save_fig:
            plt.savefig('Results/' + str(self.file) + '/' + str(title) + '.pdf')

        plt.show()

        return
