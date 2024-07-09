import numpy as np

from ..Plot.plot import SimComparePlot

# THIS SETUP IS TO look at cp/kp changes in cap/substrate:

thin_thick_files = ['CGT Paper/15nm_fl_0.5_pristine', 'CGT Paper/match_sim_thin', 'CGT Paper/TBC_100']

compare_plotter_thinthick = SimComparePlot(thin_thick_files)

compare_plotter_thinthick.kerr_plot(pen_dep=15e-9, layer_thickness=2.0e-9, min_time=0, max_time=5000, save_fig=False,
                                     filename='thin_thick_compare', norm=False)
