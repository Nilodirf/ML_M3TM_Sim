from plot import SimPlot

plotter = SimPlot(file='try_1')

plotter.map_plot(key='mag', save_fig=False)

plotter.line_plot(key='tp', min_layer=0, max_layer=10, save_fig=False)