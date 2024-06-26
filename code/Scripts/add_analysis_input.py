from ..Plot.additional_analysis import SimAnalysis

# this here plots one simulation like Fig. 1 in the document for review-response
# SimAnalysis.plot_m_meq(file='one_data_set', num_cgt_layers=45, save_file_name='CGT/thick_flu_0.5_m_meq.pdf')

# this here plots many simulations like Fig. 4 in the document for review-response

# SimAnalysis.save_dm_dt_ft(file='D:/CGT/fluence dependence/thick_flu_0.25/', max_freq=1e9, num_freq=1000,
#                           save_path='D:/CGT/fluence dependence/FFT/thick/flu_0.25')

files = [['D:/CGT/fluence dependence/FFT/thin/flu_0.5_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thin/flu_0.4_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thin/flu_0.3_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thin/flu_0.25_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thin/flu_0.2_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thin/flu_0.15_dm_dt_fft.npy'],
         ['D:/CGT/fluence dependence/FFT/thick/flu_0.5_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thick/flu_0.4_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thick/flu_0.3_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thick/flu_0.25_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thick/flu_0.2_dm_dt_fft.npy',
          'D:/CGT/fluence dependence/FFT/thick/flu_0.15_dm_dt_fft.npy']]

SimAnalysis.compare_plot_fft_dm_dt(files=files, x_axis=[0.5, 0.4, 0.3, 0.25, 0.2, 0.15], x_label=r'fluence [mJ/cm$^2$]',
                                   save_path='D:/CGT/fluence dependence/FFT/comparison_thick_thin.pdf')

# SimAnalysis.dm_dt_fft(files=thick_to_plot, label_num=label_num, save_path='CGT/FT_thick.pdf')
