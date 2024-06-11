import numpy as np
from ..Plot.analysis import SimAnalysis
import os

# SimAnalysis.fit_mag_data('bla', 0, 1e-10)
SimAnalysis.plot_m_meq(file='CGT/paper_low_flu', num_cgt_layers=75, save_file_name='CGT_thick_lowflu_m_meq.pdf')
# SimAnalysis.show_mean_field_mag(S=1.5, Tc=65., savepath='Results/CGT Paper/m_eq.pdf')
# SimAnalysis.plot_dmdt()

## NOTE FOR ME: ADD SIMULATION DATA OF DM/DT TO THE DMT_DT MAP PLOT FOR VISUALIZATION.