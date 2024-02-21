from ..Plot.analysis import SimAnalysis

thin_thick_files = ['fit/thin', 'FGT/thin_free']

# SimAnalysis.fit_umd_data('cgt', 'cgt/monolayer')
# SimAnalysis.fit_umd_data('all', ['cgt/fit_umd', 'fgt/fit_umd', 'cri3/fit_umd'])
# SimAnalysis.fit_umd_data('cgt_long', ['cgt/thin_multilayer', 'cgt/thick_multilayer'])

# SimAnalysis.fit_phonon_decay('cgt/thin_multilayer', 8, 17, 150e-12)
SimAnalysis.fit_mag_data('cgt/multi_layer_longtime', 5e-12)
