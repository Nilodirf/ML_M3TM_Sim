import numpy as np
from ..Plot.analysis import SimAnalysis
import os
# thin_thick_files = ['fit/thin', 'FGT/thin_free']

# SimAnalysis.fit_umd_data('fgt', 'fgt/fit_umd_2')
# SimAnalysis.fit_umd_data('all', ['cgt/fit_umd', 'fgt/fit_umd', 'cri3/fit_umd'])
# SimAnalysis.fit_umd_data('cgt_long', ['cgt/thin_multilayer', 'cgt/thick_multilayer_cap'])

# SimAnalysis.fit_phonon_decay('array/FGT(16)_AIN(300)_flu_0.3', 0, 8, 10e-12, end_time=3e-9)
# p0_cgt = [0.7, 0.3, 4e-12, 1e-10, 0.2, 1e-9, 5e-10]
# p0_fgt = [0.7, 0.4, -4e-12, 1e-11, 0.2, 0.6e-10, 1e-10]
# popt_int, cv_int =SimAnalysis.fit_mag_data(file='CGT/thin_multilayer', t1=1.5e-12, t_max=2e-9, p0_initial=p0_cgt)
# popt_der, cv_der = SimAnalysis.fit_dm_dt(file='fgt/thin_mulitlayer', t1=1e-12, p0_initial=popt_int[1:])

SimAnalysis.fit_FGT_remag('fgt/high_flu', 1.5, 220.)
# SimAnalysis.mag_tem_plot('cgt/thin_multilayer')


# ferro_sub_flu_files = os.listdir('C:/Users/Theodor Griepe/Documents/Github/CGT/Results/array_cap')
# for i, string in enumerate(ferro_sub_flu_files):
#     ferro_sub_flu_files[i] = 'array_cap/' + string
# tp_analysis = SimAnalysis(ferro_sub_flu_files)
# tp_analysis.fit_all_mag('cap_array_mag')

# ferro_cap_flu_files = os.listdir('C:/Users/Theodor Griepe/Documents/Github/CGT/Results/array_cap')
# for i, string in enumerate(ferro_cap_flu_files):
#     ferro_cap_flu_files[i] = 'array_cap/' + string
# tp_analysis = SimAnalysis(ferro_cap_flu_files)
# tp_analysis.fit_all_phonon('cap_array_phonons', 150, 158, None)
#
# ferro_sub_flu_files = os.listdir('C:/Users/Theodor Griepe/Documents/Github/CGT/Results/array')
# for i, string in enumerate(ferro_sub_flu_files):
#     ferro_sub_flu_files[i] = 'array/' + string
# tp_analysis = SimAnalysis(ferro_sub_flu_files)
# tp_analysis.fit_all_phonon('sub_array_phonons', 0, 8, None)
#
# SimAnalysis.plot_phonon_params('cap_array_phonons.npy', 'cap_phonon_fits')
# SimAnalysis.plot_phonon_params('sub_array_phonons.npy', 'sub_phonon_fits')

