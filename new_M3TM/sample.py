import numpy as np

class sim_sample:
    # Within this class the sample as a 1d-array of materials can be constructed and parameters and functions for
    # the whole sample can be defined and read out.
    def __init__(self):
        self.mat_arr = np.array([])

    def add_layers(self, material, layers):
        # This method lets us add layers to the sample

        # Input:
        # self (object). A pointer to the sample in construction
        # material (object). A material previously defined with the materials class
        # layers (int). Number of layers with depth material.dz to be added to the sample

        # Returns:
        # mat_arr (numpy array). 1d-array after the layers have been added
        self.mat_arr = np.append(self.mat_arr, np.array([material for _ in range(layers)]))
        return self.mat_arr

    def get_params(self, param):
        # This method lets us read out the parameters of all layers in the sample

        # Input:
        # self (object). A pointer to the sample in use
        # param (String). String of the paramter defined in the material class to be read out

        # Returns:
        # params (numpy array). 1d-array of the parameters asked for
        return np.array([mat.__dict__[param] for mat in self.mat_arr])

    def get_len(self):
        # This method merely counts the number of layers and returns it

        # Input:
        # self (object). A pointer to the sample in use

        # Returns:
        # len(sample) (int). Number of layers in the sample
        return len(self.mat_arr)

    def get_material_changes(self):
        # This method divides the sample into blocks consisting of the same material

        # Input:
        # self (object): The sample in use

        # Returns:
        # material_blocks (list). List of the blocks of layers separated by changes in materials
        # along the sample, containing the number of repeated layers

        material_blocks = []
        n_sam = self.get_len()
        same_mat_counter=1

        for i in range(1, n_sam):
            if self.mat_arr[i] == self.mat_arr[i-1]:
                same_mat_counter+=1
            else:
                material_blocks.append(same_mat_counter)
                same_mat_counter=0

        return material_blocks


sample = sim_sample()
# The following wont work here as I did not import mats.py
sample.add_layers(CGT, 10)
sample.add_layers(sio2, 3)
