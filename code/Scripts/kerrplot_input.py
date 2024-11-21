import numpy as np

from ..Plot.plot import SimComparePlot

e_term_files = ['therm_time_tests/0.0', 'therm_time_tests/2e-14', 'therm_time_tests/3e-14', 'therm_time_tests/4e-14',
                'therm_time_tests/5e-14', 'therm_time_tests/6e-14', 'therm_time_tests/7e-14', 'therm_time_tests/8e-14',
                'therm_time_tests/9e-14']

compare_plotter = SimComparePlot(e_term_files)

compare_plotter.compare_sims(key='te', min_layers=[0 for i in e_term_files], max_layers=[1 for i in e_term_files])
