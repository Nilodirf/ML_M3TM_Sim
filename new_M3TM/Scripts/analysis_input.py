import numpy as np
from ..Plot.analysis import SimAnalysis

thin_thick_files = ['fit/thin', 'FGT/thin_free']

# SimAnalysis.fit_umd_data('cgt', 'cgt/monolayer')
# SimAnalysis.fit_umd_data('all', ['cgt/fit_umd', 'fgt/fit_umd', 'cri3/fit_umd'])
# SimAnalysis.fit_umd_data('cgt_long', ['cgt/thin_multilayer', 'cgt/thick_multilayer'])

# SimAnalysis.fit_phonon_decay('cgt/thin_multilayer', 8, 17, 5e-12)
popt_int, cv_int =SimAnalysis.fit_mag_data('fgt/thin_mulitlayer_nocap', 1.5e-12)
popt_der, cv_der = SimAnalysis.fit_dm_dt('fgt/thin_mulitlayer', 4e-12, popt_int[1:])

