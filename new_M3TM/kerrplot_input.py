from plot import SimComparePlot

# Initialize the plot class with the simulation results folder denoted 'files':
compare_plotter = SimComparePlot(['15_exact', '150_exact'])

# Plot the Kerr-signal for all files in one plot, denoting the penetration depth and layer thickness of the material.
compare_plotter.kerr_plot(pen_dep=30e-9, layer_thickness=2.05e-9, min_time=0, max_time=3000, save_fig=False)
