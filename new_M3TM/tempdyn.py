import numpy as np

class temperature_sim:
    # This class holds all methods needed for temperature calculations.

    def initialize(self, sample, initemp):
        # This method initializes the starting uniform temperature map.

        # Input:
        # self (object). The temperature class
        # sample (object). The sample in use
        # initemp (float). The starting temperature in K

        # Output:
        # te_arr (numpy array). 1d-array of the starting electron temperatures
        # tp_arr (numpy array). 1d-array of the starting phonon temperatures

        n_sam = sample.get_len()
        te_arr = np.array()