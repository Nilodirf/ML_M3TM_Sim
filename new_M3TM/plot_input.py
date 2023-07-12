# Here you can call some plot functions, plot and save plots

from plot import SimPlot


# Initialize the plot class with the simulation results folder denoted 'file':
plotter = SimPlot(file='150_nm_hbn_abs_high')

# Plot a map of the denoted simulation of one of the three subsystems, save if you want to:
plotter.map_plot(key='tp', min_layer=14, max_layer=40, min_time=0, max_time=10, save_fig=False)
plotter.map_plot(key='te', min_time=0, max_time=10, save_fig=False)

# Plot the dynamics of one subsystem for some layers in line-plots to see the dynamics, save if you want to:
plotter.line_plot(key='te', max_layer=30, average=False, save_fig=False)
plotter.line_plot(key='mag', min_layer=0, max_layer=2, average=True, save_fig=True)
