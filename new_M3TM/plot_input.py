# Here you can call some plot functions, plot and save plots

from plot import SimPlot


# Initialize the plot class with the simulation results folder denoted 'file':
plotter = SimPlot(file='150_nm')

# Plot a map of the denoted simulation of one of the three subsystems, save if you want to:
# plotter.map_plot(key='tp', max_layer=30, max_time=30, save_fig=True)
# plotter.map_plot(key='te', max_time=30, save_fig=True)

# Plot the dynamics of one subsystem for some layers in line-plots to see the dynamics, save if you want to:
# plotter.line_plot(key='te', average=False, min_layer=0, max_layer=30, save_fig=True)
# plotter.line_plot(key='tp', average=False, min_layer=0, max_layer=30, save_fig=True)
plotter.line_plot(key='tp', min_layer=0, max_layer=17, average=False, save_fig=True)
