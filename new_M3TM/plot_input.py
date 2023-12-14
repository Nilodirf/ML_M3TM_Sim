# Here you can call some plot functions, plot and save plots

from plot import SimPlot


# Initialize the plot class with the simulation results folder denoted 'file':
plotter = SimPlot(file='try_fit')

# plotter.convert_to_dat()

# Plot a map of the denoted simulation of one of the three subsystems, save if you want to:
# plotter.map_plot(key='tp',  save_fig=False)
# plotter.map_plot(key='tp', kind='surface', max_time=3000, max_layer=133, save_fig=True, filename='thick_alltime_tp')

# Plot the dynamics of one subsystem for some layers in line-plots to see the dynamics, save if you want to:
# plotter.line_plot(key='tp', min_layer=0, max_layer=19, average=False, save_fig=False)
# plotter.line_plot(key='te', min_layer=0, max_layer=4, average=True, save_fig=False, norm=False)
plotter.line_plot(key='mag', min_layer=0, max_layer=4, average=False, save_fig=False, norm=False)
