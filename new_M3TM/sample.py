import numpy as np

class SimSample:
    # Within this class the sample as a 1d-array of materials can be constructed and parameters and functions for
    # the whole sample can be defined and read out.

    def __init__(self):
        self.mat_arr = np.array([])
        self.len = self.get_len()
        self.mat_blocks = self.get_material_changes()
        self.el_mask = self.get_free_electron_mask()
        self.magdyn_mask = self.get_magdyn_mask()
        self.mats, self.mat_ind = self.get_mat_positions()
        self.mag_num = self.get_num_mag_mat()

    def add_layers(self, material, layers):
        # This method lets us add layers to the sample. It also automatically recalculates other sample properties.

        # Input:
        # self (object). A pointer to the sample in construction
        # material (object). A material previously defined with the materials class
        # layers (int). Number of layers with depth material.dz to be added to the sample

        # Returns:
        # mat_arr (numpy array). 1d-array after the layers have been added

        self.mat_arr = np.append(self.mat_arr, np.array([material for _ in range(layers)]))
        self.len = self.get_len()
        self.mat_blocks = self.get_material_changes()
        self.el_mask = self.get_free_electron_mask()
        self.mats, self.mat_ind = self.get_mat_positions()
        self.mag_num = self.get_num_mag_mat()

        return self.mat_arr

    def get_params(self, param):
        # This method lets us read out the parameters of all layers in the sample

        # Input:
        # self (object). A pointer to the sample in use
        # param (String). String of the parameter defined in the material class to be read out

        # Returns:
        # params (numpy array). 1d-array of the parameters asked for

        if param == 'cp_T':
            return np.array([mat.cp_T_grid for mat in self.mats]), np.array([mat.cp_T for mat in self.mats])
        else:
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
        same_mat_counter = 1

        for i in range(1, n_sam):
            if self.mat_arr[i] == self.mat_arr[i-1]:
                same_mat_counter += 1
            else:
                material_blocks.append(same_mat_counter)
                same_mat_counter = 1
        material_blocks.append(same_mat_counter)

        return material_blocks

    def initialize_temperature(self, ini_temp):
        # This method initializes the starting uniform temperature map.

        # Input:
        # self (object). The sample in use
        # sample (object). Initial starting temperature

        # Returns:
        # te_arr (numpy array). 1d-array of the starting electron temperatures
        # tp_arr (numpy array). 1d-array of the starting phonon temperatures

        te_arr = np.ones_like(self.mat_arr)*ini_temp
        tp_arr = te_arr.copy()

        return te_arr, tp_arr

    def get_free_electron_mask(self):
        # This method selects all layers in the sample where the electron dynamics need to be computed.
        # If there are no conduction electrons, there shall be no pulse absorption
        # no electron temperature or diffusion.

        # Input:
        # self (object). The sample in use

        # Returns:
        # free_electron_mask (boolean array). 1d-array holding True for all layers where gamma_e!=0

        free_electron_mask = self.get_params('ce_gamma') != 0

        return free_electron_mask

    def get_magdyn_mask(self):
        # This method selects the layers where magnetization dynamics need to be computed. we select this with the
        # parameter muat.

        # Input:
        # self (object). The sample in use

        # Returns:
        # mag_dyn_mask (boolean array). 1d-array selecting the layers where magnetization dynamics shall be computed

        magdyn_mask = self.get_params('muat') != 0

        return magdyn_mask

    def get_mat_positions(self):
        mats = []
        for mat in list(self.mat_arr):
            if mat not in mats:
                mats.append(mat)

        mat_indices = [[] for _ in mats]
        for j, mat in enumerate(mats):
            for i in range(self.get_len()):
                if self.mat_arr[i] == mat:
                    mat_indices[j].append(i)

        return mats, [np.array(ind_list) for ind_list in mat_indices]

    def get_num_mag_mat(self):
        mag_counter = 0
        for mat in self.mat_arr:
            if mat.muat != 0:
                mag_counter += 1

        return mag_counter
