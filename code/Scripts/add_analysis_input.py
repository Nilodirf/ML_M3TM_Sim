from ..Plot.additional_analysis import SimAnalysis

# this here plots one simulation like Fig. 1 in the document for review-response
# this here plots many simulations like Fig. 4 in the document for review-response

fft_files = [['E:/CGT/fluence dependence/FFT/thin_flu_0.5.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.45.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.4.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.35.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.3.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.25.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.2.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.15.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.1.npy',
              'E:/CGT/fluence dependence/FFT/thin_flu_0.05.npy'],
             ['E:/CGT/fluence dependence/FFT/thick_flu_0.5.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.45.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.4.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.35.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.3.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.25.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.2.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.15.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.1.npy',
              'E:/CGT/fluence dependence/FFT/thick_flu_0.05.npy']]

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

files_on_work_pc = [['Results/CGT/fluence dependence/thin_flu_0.5',
                     'Results/CGT/fluence dependence/thin_flu_0.45',
                     'Results/CGT/fluence dependence/thin_flu_0.4',
                     'Results/CGT/fluence dependence/thin_flu_0.35',
                     'Results/CGT/fluence dependence/thin_flu_0.3',
                     'Results/CGT/fluence dependence/thin_flu_0.25',
                     'Results/CGT/fluence dependence/thin_flu_0.2',
                     'Results/CGT/fluence dependence/thin_flu_0.15',
                     'Results/CGT/fluence dependence/thin_flu_0.1',
                     'Results/CGT/fluence dependence/thin_flu_0.05'],
                    ['Results/CGT/fluence dependence/thick_flu_0.5',
                     'Results/CGT/fluence dependence/thick_flu_0.45',
                     'Results/CGT/fluence dependence/thick_flu_0.4',
                     'Results/CGT/fluence dependence/thick_flu_0.35',
                     'Results/CGT/fluence dependence/thick_flu_0.3',
                     'Results/CGT/fluence dependence/thick_flu_0.25',
                     'Results/CGT/fluence dependence/thick_flu_0.2',
                     'Results/CGT/fluence dependence/thick_flu_0.15',
                     'Results/CGT/fluence dependence/thick_flu_0.1',
                     'Results/CGT/fluence dependence/thick_flu_0.05']]

fft_files_on_work_pc = [['Results/CGT/fluence dependence/FFT/thin_flu_0.5.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.45.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.4.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.35.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.3.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.25.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.2.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.15.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.1.npy',
                         'Results/CGT/fluence dependence/FFT/thin_flu_0.05.npy'],
                        ['Results/CGT/fluence dependence/FFT/thick_flu_0.5.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.45.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.4.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.35.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.3.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.25.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.2.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.15.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.1.npy',
                         'Results/CGT/fluence dependence/FFT/thick_flu_0.05.npy']]

# for file in files_on_work_pc[1]:
#     SimAnalysis.save_dm_dt_ft(file=file, max_freq=0.5e9, num_freq=2000, save_path='Results/CGT/fluence dependence/FFT/' + str(file).replace('Results/CGT/fluence dependence/', ''))

# SimAnalysis.plot_spin_acc(files=files_on_work_pc[1], labels=['0.5', '0.45', '0.4', '0.35', '0.3', '0.25', '0.2', '0.15', '0.1', '0.05'], save_path='CGT/fluence dependence/mus_flu_thick.pdf')

# SimAnalysis.plot_fft_dm_dt(files=fft_files_on_work_pc[1], x_axis=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
#                            x_label=r'fluence [mJ/cm$^2$]', save_path='Results/CGT/fluence dependence/FFT/FT_thick_log.pdf')

SimAnalysis.plot_max_magrate_distance(files=files_on_work_pc[1], x_axis=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05], x_label=r'fluence [mJ/cm$^2$]', save_path='Results/CGT/fluence dependence/time_diff_thick_log.pdf')

# SimAnalysis.dm_dt_fft(files=thick_to_plot, label_num=label_num, sa
# ve_path='CGT/FT_thick.pdf')
