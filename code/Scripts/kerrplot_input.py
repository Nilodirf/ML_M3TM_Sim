import numpy as np

from ..Plot.plot import SimComparePlot

e_term_files = ["low_kapitza", "high_kapitza"]

compare_plotter = SimComparePlot(e_term_files)

compare_plotter.compare_sims(key='mag', min_layers=[0, 0], max_layers=[69, 69])
