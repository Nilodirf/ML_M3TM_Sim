import numpy as np
from ..Plot.analysis import SimAnalysis

thin_thick_files = ['fit/thin', 'FGT/thin_free']

# SimAnalysis.fit_umd_data('fgt', 'fgt/fit_umd')
# SimAnalysis.fit_umd_data('all', ['cgt/fit_umd', 'fgt/fit_umd', 'cri3/fit_umd'])
# SimAnalysis.fit_umd_data('cgt_long', ['cgt/thin_multilayer', 'cgt/thick_multilayer'])

# SimAnalysis.fit_phonon_decay('cgt/thin_multilayer', 8, 17, 5e-12)
# p0_cgt = [0.7, 0.3, 4e-12, 1e-10, 0.2, 1e-9, 5e-10]
# p0_fgt = [0.7, 0.4, -4e-12, 1e-11, 0.2, 0.6e-10, 1e-10]
# popt_int, cv_int =SimAnalysis.fit_mag_data(file='FGT/sub_AIN', t1=5e-12, t_max=0.5e-9, p0_initial=p0_fgt)
# popt_der, cv_der = SimAnalysis.fit_dm_dt(file='fgt/thin_mulitlayer', t1=1e-12, p0_initial=popt_int[1:])

SimAnalysis.fit_mag_tau1(file='cgt/thin_multilayer', kind='LLB', show_fig=True, save_fig=False, t1=10e-12)
SimAnalysis.fit_mag_tau2(file='cgt/thin_multilayer', kind='exp', show_fig=True, save_fig=False)

