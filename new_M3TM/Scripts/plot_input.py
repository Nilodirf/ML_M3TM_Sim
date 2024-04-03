# Here you can call some plot functions, plot and save plots

from ..Plot.plot import SimPlot


# Initialize the plot class with the simulation results folder denoted 'file':
plotter = SimPlot(file='tim_all_ke')

# plotter.convert_to_dat()

# Plot a map of the denoted simulation of one of the three subsystems, save if you want to:
# plotter.map_plot(key='tp',  save_fig=False)

# plotter.map_plot(key='mag', kind='surface', max_time=2000, save_fig=False, filename='thin_alltime_mag', show_title=False, color_scale='inferno', vmin=0, vmax=1)
plotter.map_plot(key='te', kind='colormap', save_fig=False, show_title=False, color_scale='Reds_r')
plotter.map_plot(key='tp', kind='colormap', save_fig=False, show_title=False, color_scale='Blues_r')


# Plot the dynamics of one subsystem for some layers in line-plots to see the dynamics, save if you want to:
# plotter.line_plot(key='tp', average=False, save_fig=False)
plotter.line_plot(key='te', average=False, save_fig=False, norm=False)

# plotter.line_plot(key='mag', average=False, save_fig=False, norm=False)

# plotter.te_tp_plot(max_time=50, average=False, save_fig=False,  filename='te_tp_thick', tp_layers=[0, 16], color_scales=['Greens_r', 'Blues_r'])  #8,84 for thick, 8,
