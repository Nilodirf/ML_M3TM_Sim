import numpy as np
import os
from matplotlib import pyplot as plt


class FGTSim:

    def __init__(self, file):
        self.file = file

        return

    def get_data(self):

        sim_files_folder = 'Results/FGT/' + self.file + '/'
        delay = np.load(sim_files_folder + 'delays.npy')
        te = np.load(sim_files_folder + 'tes.npy')
        tp = np.load(sim_files_folder + 'tps.npy')
        mag = np.load(sim_files_folder + 'ms.npy')

        if os.path.isfile(sim_files_folder + 'tp2s.npy'):
            tp2 = np.load(sim_files_folder + 'tp2s.npy')
        else:
            tp2 = np.zeros_like(tp)