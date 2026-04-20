import numpy as np
from ..Plot.analysis import SimAnalysis
import os

# SimAnalysis.fit_mag_data('bla', 0, 1e-10)
# SimAnalysis.plot_m_meq(file='CGT Paper/90nm_fl_0.5_pristine', num_cgt_layers=45, save_file_name='CGT/thick_flu_0.5_m_meq.pdf')
# SimAnalysis.plot_mean_field_mag(S=1.5, Tc=65., savepath='Results/CGT/mu_s.pdf')
# SimAnalysis.plot_dmdt()

files_thin = ['CGT Paper/15nm_fl_0.5_pristine', 'CGT/fluence dependence/thin_flu_0.4',
              'CGT/fluence dependence/thin_flu_0.3', 'CGT/fluence dependence/thin_flu_0.2',
              'CGT/fluence dependence/thin_flu_0.1']

files_thick = ['CGT Paper/90nm_fl_0.5_pristine', 'CGT/fluence dependence/thick_flu_0.4',
               'CGT/fluence dependence/thick_flu_0.3', 'CGT/fluence dependence/thick_flu_0.2',
               'CGT/fluence dependence/thick_flu_0.1']

SimAnalysis(files=files_thick).plot_spin_acc(S=1.5, Tc=65., save_path='CGT/fluence dependence/all_mu_thick.pdf')
