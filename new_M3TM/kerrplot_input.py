import numpy as np

from plot import SimComparePlot

cp_string = 'sub/thin_CGT/cp_1800000.0_kp_'
kp_string_list = np.arange(0, 11, dtype=float).astype(str)

kp_string = '_kp_1.0'
cp_string_default = 'sub/thin_CGT/cp_'
cp_string_list = ['1500000.0', '1800000.0', '2100000.0', '2400000.0', '2700000.0', '3000000.0']

# Initialize the plot class with the simulation results folder denoted 'files':
compare_plotter_kp = SimComparePlot([cp_string + i for i in kp_string_list])
compare_plotter_cp = SimComparePlot([cp_string_default + i + kp_string for i in cp_string_list])

# Plot the Kerr-signal for all files in one plot, denoting the penetration depth and layer thickness of the material.
compare_plotter_kp.kerr_plot(pen_dep=30e-9, layer_thickness=2.0e-9, min_time=0, max_time=3000, save_fig=False,
                             filename='different_kp', norm=False)
compare_plotter_cp.kerr_plot(pen_dep=30e-9, layer_thickness=2.0e-9, min_time=0, max_time=3000, save_fig=False,
                             filename='different_cp', norm=False)
