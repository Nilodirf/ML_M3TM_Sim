# Here you can call some plot functions, plot and save plots

from ..Plot.plot import SimPlot
from ..Plot.plot import SimComparePlot


# Initialize the plot class with the simulation results folder denoted 'file':
plotter = SimPlot(file='CGT Paper/TBC_100')

# plotter.convert_to_dat()

# Plot a map of the denoted simulation of one of the three subsystems, save if you want to:
# plotter.map_plot(key='tp',  save_fig=False)

# plotter.map_plot(key='mag', kind='surface', max_time=2000, save_fig=False, filename='tutorial_mag', show_title=False, color_scale='inferno', vmin=0, vmax=1)
plotter.map_plot(key='te', kind='surface', save_fig=False, filename='fullsam_te', show_title=False, color_scale='Reds')
plotter.map_plot(key='tp', kind='surface', save_fig=False, filename='fullsam_tp', show_title=False, color_scale='Blues')


# Plot the dynamics of one subsystem for some layers in line-plots to see the dynamics, save if you want to:
plotter.line_plot(key='tp', average=False, save_fig=False)
# plotter.line_plot(key='te', min_layer=0, max_layer=1, average=False, save_fig=False, norm=False)
# plotter.line_plot(key='tp', min_layer=0, max_layer=1, average=False, save_fig=False, norm=False)

# plotter.line_plot(key='mag', average=False, save_fig=False, norm=False)

# plotter.te_tp_plot(max_time=6.5, average=False, save_fig=False,  filename='te_tp_thick', tp_layers=[2, 12], color_scales=['Greens_r', 'Blues_r'])  #8,84 for thick, 8,

# tim_fu_files = ['FU_Tim_Terbium/simple_sample', 'FU_Tim_Terbium/full_sample']
# SimComparePlot(tim_fu_files).compare_samples(key='te', min_layers=[2, 2], max_layers=[12, 12], colors=['orange', 'green'],
#                                              labels=['only Tb Abeles', 'full sample Abeles'], save_fig=True,
#                                              filename='FU_Tim_Terbium/te_comp')
