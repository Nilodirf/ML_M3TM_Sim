from ..Plot.additional_analysis import SimAnalysis

# this here plots one simulation like Fig. 1 in the document for review-response
# SimAnalysis.plot_m_meq(file='one_data_set', num_cgt_layers=45, save_file_name='CGT/thick_flu_0.5_m_meq.pdf')

# this here plots many simulations like Fig. 4 in the document for review-response

# SimAnalysis.save_dm_dt_ft(file='E:/CGT/fluence dependence/thick_flu_0.1/', max_freq=1e9, num_freq=2000,
#                           save_path='E:/CGT/fluence dependence/FFT/thick/flu_0.1')

fft_files = [['E:/CGT/fluence dependence/FFT/thin/flu_0.5.npy',
          'E:/CGT/fluence dependence/FFT/thin/flu_0.45.npy',
          'E:/CGT/fluence dependence/FFT/thin/flu_0.4.npy',
          'E:/CGT/fluence dependence/FFT/thin/flu_0.35.npy',
          'E:/CGT/fluence dependence/FFT/thin/flu_0.3.npy',
          'E:/CGT/fluence dependence/FFT/thin/flu_0.25.npy',
          'E:/CGT/fluence dependence/FFT/thin/flu_0.2.npy',
          'E:/CGT/fluence dependence/FFT/thin/flu_0.15.npy',
          'E:/CGT/fluence dependence/FFT/thin/flu_0.1.npy'],
         ['E:/CGT/fluence dependence/FFT/thick/flu_0.5.npy',
          'E:/CGT/fluence dependence/FFT/thick/flu_0.45.npy',
          'E:/CGT/fluence dependence/FFT/thick/flu_0.4.npy',
          'E:/CGT/fluence dependence/FFT/thick/flu_0.35.npy',
          'E:/CGT/fluence dependence/FFT/thick/flu_0.3.npy',
          'E:/CGT/fluence dependence/FFT/thick/flu_0.25.npy',
          'E:/CGT/fluence dependence/FFT/thick/flu_0.2.npy',
          'E:/CGT/fluence dependence/FFT/thick/flu_0.15.npy',
          'E:/CGT/fluence dependence/FFT/thick/flu_0.1.npy']]

files = [['E:/CGT/fluence dependence/thin_flu_0.5',
          'E:/CGT/fluence dependence/thin_flu_0.45',
          'E:/CGT/fluence dependence/thin_flu_0.4',
          'E:/CGT/fluence dependence/thin_flu_0.35',
          'E:/CGT/fluence dependence/thin_flu_0.3',
          'E:/CGT/fluence dependence/thin_flu_0.25',
          'E:/CGT/fluence dependence/thin_flu_0.2',
          'E:/CGT/fluence dependence/thin_flu_0.15',
          'E:/CGT/fluence dependence/thin_flu_0.1'],
         ['E:/CGT/fluence dependence/thick_flu_0.5',
          'E:/CGT/fluence dependence/thick_flu_0.45',
          'E:/CGT/fluence dependence/thick_flu_0.4',
          'E:/CGT/fluence dependence/thick_flu_0.35',
          'E:/CGT/fluence dependence/thick_flu_0.3',
          'E:/CGT/fluence dependence/thick_flu_0.25',
          'E:/CGT/fluence dependence/thick_flu_0.2',
          'E:/CGT/fluence dependence/thick_flu_0.15',
          'E:/CGT/fluence dependence/thick_flu_0.1']]

SimAnalysis.plot_spin_acc(files=files[1], labels=['0.5', '0.45', '0.4', '0.35', '0.3', '0.25', '0.2', '0.15', '0.1'], save_path = 'mus_flu_thick_2.pdf')

# SimAnalysis.plot_fft_dm_dt(files=fft_files[1], x_axis=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1], x_label=r'fluence [mJ/cm$^2$]', save_path='E:/CGT/figs/FT_thick.pdf')

# SimAnalysis.plot_max_magrate_distance(files=files[0], x_axis=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1], x_label=r'fluence [mJ/cm$^2$]', save_path='E:/CGT/figs/time_diff_thin.pdf')

# SimAnalysis.dm_dt_fft(files=thick_to_plot, label_num=label_num, save_path='CGT/FT_thick.pdf')
