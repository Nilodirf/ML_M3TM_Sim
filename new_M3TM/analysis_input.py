from analysis import SimAnalysis
import numpy as np

thin_thick_files = ['fit/thin', 'FGT/thin_free']

cp_string = 'sub/thin_CGT/cp_1500000.0_kp_'
kp_string_list = np.arange(1, 11, dtype=float).astype(str)
anal_kp = SimAnalysis([cp_string + i for i in kp_string_list])

anal_15nm = SimAnalysis(thin_thick_files)

# anal_kp.mag_tem_plot()
anal_15nm.plot_dmdt()
