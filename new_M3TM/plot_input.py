# Here you can call some plot functions, plot and save plots

from plot import SimPlot


# Initialize the plot class with the simulation results folder denoted 'file':
plotter = SimPlot(file='fit/thick_kp2')

# plotter.convert_to_dat()

# Plot a map of the denoted simulation of one of the three subsystems, save if you want to:
# plotter.map_plot(key='tp',  save_fig=False)
<<<<<<< Updated upstream
# plotter.map_plot(key='mag', kind='surface', max_time=3000, max_layer=65, save_fig=True, filename='thin_alltime_mag', show_title=False, color_scale='summer')
=======
# plotter.map_plot(key='mag', kind='surface', max_time=3000, max_layer=65, save_fig=True, filename='thin_alltime_mag', show_title=False)
>>>>>>> Stashed changes
plotter.map_plot(key='tp', kind='surface', max_time=3000, save_fig=False, filename='thick_alltime_mag', show_title=False)


# Plot the dynamics of one subsystem for some layers in line-plots to see the dynamics, save if you want to:
# plotter.line_plot(key='tp', min_layer=16, average=True, save_fig=False)
# plotter.line_plot(key='te', min_layer=0, max_layer=4, average=True, save_fig=False, norm=False)
<<<<<<< Updated upstream
plotter.line_plot(key='mag', min_layer=0, max_layer=4, average=False, save_fig=False, norm=False)
=======
# plotter.line_plot(key='mag', min_layer=0, max_layer=4, average=True, save_fig=False, norm=False)
>>>>>>> Stashed changes
# plotter.te_tp_plot(max_time=6, tp_layers=[9, 17])
