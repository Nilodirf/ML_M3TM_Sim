import numpy as np

from plot import SimComparePlot

cp_string = 'sub/cp_1800000.0_kp_'
kp_string = np.arange(0, 11, dtype=float).astype(str)

# Initialize the plot class with the simulation results folder denoted 'files':
compare_plotter = SimComparePlot([cp_string+i for i in kp_string])

# Plot the Kerr-signal for all files in one plot, denoting the penetration depth and layer thickness of the material.
compare_plotter.kerr_plot(pen_dep=30e-9, layer_thickness=2.0e-9, min_time=0, max_time=3000, save_fig=False, norm=False)
