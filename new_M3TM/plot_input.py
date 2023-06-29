from plot import SimPlot
# Here you can call some plot functions, plot and save plots

# Initialize the plot class with the simulation results folder denoted 'file':
plotter = SimPlot(file='try_1')

# Plot a map of the denoted simulation of one of the three subsystems, save if you want to:
plotter.map_plot(key='mag', save_fig=False)

# Plot the dynamics of one subsystem for some layers in line-plots to see the dynamics, save if you want to:
plotter.line_plot(key='tp', min_layer=0, max_layer=10, save_fig=False)
