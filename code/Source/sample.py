import numpy as np


class SimSample:
    # Within this class the sample as a 1d-array of materials can be constructed and parameters and functions for
    # the whole sample can be defined and read out.

    def __init__(self):
        # Input:

        # Also returns (all recalculated after addition of layers with self.add_layers):
        # mat_arr (numpy array) 1d-array of the materials at their respective positions in the sample
        # len (int). Number of layers in the sample
        # mat_blocks (list of lists). Containing the number of subsequent layers of the same material in each sub_list
        # el_mask (boolean array). Mask showing at which position materials with itinerant electrons are placed
        # mag_mask (boolean array). Mask showing at which position magnetic materials are placed
        # tp2_mask (boolean array). Mask showing at which positions a second phononic system is to be considered
        # mats, mat_ind (list, list). List of the different materials in the sample and their positions
        # mag_num (int). Number of magnetic materials in the sample.
        # kappa_p_int (numpy array). 1d-array of the interface constants of phononic heat diffusion. Starts empty,
        # recalculated after adding of layers
        # kappa_e_int (numpy array). 1d-array of the interface constants of electronic heat diffusion. Starts empty,
        # recalculated after adding of layers
        # len_te (int). Number of layers that have free electrons (determined with el_mask)
        # len_tp2 (int). Numer of layers that have free electrons (determined with tp2_mask)
        # el_mag_mask (boolean array). Array of the length of numbers of layers that hold free electrons, True if
        # also magnetic, False if not
        # n_comp_arr (numpy array). 1d-array of the complex refractive indices of the sample constituents (blocks)
        # pen_dep_arr (numpy array). 1d-array of the penetration depths of sample consituents (blocks)

        self.mat_arr = np.array([])
        self.len = self.get_len()
        self.mat_blocks = self.get_material_changes()
        self.el_mask = self.get_free_electron_mask()
        self.mag_mask = self.get_magdyn_mask()
        self.tp2_mask = self.get_tp2_mask()
        self.mats, self.mat_ind = np.array([]), np.array([])
        self.mag_num = self.get_num_mag_mat()
        self.kappa_p_int = np.array([])
        self.kappa_e_int = np.array([])
        self.len_te = int(np.sum(np.ones(self.len)[self.el_mask]))
        self.len_tp2 = int(np.sum(np.ones(self.len)[self.tp2_mask]))
        self.el_mag_mask = self.get_el_mag_mask()
        self.n_comp_arr = np.array([])
        self.pen_dep_arr = np.array([])
        self.dz_arr = np.array([])
        self.constituents = np.array([])
        self.mat_tp2_ind = []

    def add_layers(self, material, dz, layers, kappap_int=None, kappae_int=None, n_comp=None, pen_dep=None):
        # This method lets us add layers to the sample. It also automatically recalculates other sample properties.

        # Input:
        # self (object). A pointer to the sample in construction
        # material (object). A material previously defined with the materials class
        # dz (float). Layer thickness of the material in m. Important only for resolution of heat diffusion
        # layers (int). Number of layers with depth material.dz to be added to the sample
        # kappap_int (float/string). Phononic thermal interface conductance at the interface
        # to the previously added material in W/m^2/K
        # kappae_int (float/string). Electronic thermal interface conductance at the interface
        # to the previously added material in W/m^2/K
        # n_comp (complex float). Complex refractive index of the material. Use syntax 'n_r'+'n_i'j to initiate.

        # Returns:
        # mat_arr (numpy array). 1d-array after the layers have been added

        assert n_comp is not None or pen_dep is not None, ('Please introduce either a complex refraction index '
                                                           'or penetration depth for the computation of the pulse '
                                                           'profile.')

        if self.len > 0:
            assert kappap_int is not None and \
                   (type(kappap_int) == float), 'Introduce phononic diffusion interface constant ' \
                                                'using kappap_int = <value> (in MW/m^2/K)'

            if material.ce_gamma != 0 and self.mat_arr[-1].ce_gamma != 0:
                assert kappae_int is not None and \
                       (type(kappae_int) == float), 'Introduce electronic diffusion interface constant ' \
                                                    'using kappap_int = <value> (in MW/m^2/K) '

            else:
                kappae_int = 0.

            self.kappa_p_int = np.append(self.kappa_p_int, kappap_int*dz*1e6)
            self.kappa_e_int = np.append(self.kappa_e_int, kappae_int*dz*1e6)

        self.mat_arr = np.append(self.mat_arr, np.array([material for _ in range(layers)]))
        self.len = self.get_len()
        self.mat_blocks = self.get_material_changes()
        self.el_mask = self.get_free_electron_mask()
        self.mag_mask = self.get_magdyn_mask()
        self.tp2_mask = self.get_tp2_mask()
        self.mats, self.mat_ind = self.get_mat_positions()
        self.mag_num = self.get_num_mag_mat()
        self.len_te = int(np.sum(np.ones(self.len)[self.el_mask]))
        self.len_tp2 = int(np.sum(np.ones(self.len)[self.tp2_mask]))
        self.el_mag_mask = self.get_el_mag_mask()
        self.n_comp_arr = np.append(self.n_comp_arr, n_comp)
        self.pen_dep_arr = np.append(self.pen_dep_arr, pen_dep)
        self.dz_arr = np.append(self.dz_arr, dz)
        self.constituents = np.append(self.constituents, material.name)
        self. mat_tp2_ind = self.get_mat_tp2_positions()

        return self.mat_arr

    def get_params_from_blocks(self, param):
        # This method returns parameters defined for blocks of the sample for each block of layers within this class
        # and returns the same values projected onto a 1d array for all layers individually.

        # Input:
        # self (object). The sample object in use
        # param (Str). The parameter to be given for all layers of the sample

        # Returns:
        # params (numpy array). 1d-array of the parameters requested for all layers in the sample

        if param == 'pen_dep':
            return np.concatenate(np.array(
                [[self.pen_dep_arr[i] for _ in range(self.mat_blocks[i])] for i in range(len(self.mat_blocks))],
                dtype=object))

        elif param == 'n_comp':
            return np.concatenate(np.array(
                [[self.n_comp_arr[i] for _ in range(self.mat_blocks[i])] for i in range(len(self.mat_blocks))],
                dtype=object))

        elif param == 'dz':
            return np.concatenate(np.array(
                [[self.dz_arr[i] for _ in range(self.mat_blocks[i])] for i in range(len(self.mat_blocks))],
                dtype=object))

    def get_params(self, param):
        # This method lets us read out the parameters of all layers in the sample

        # Input:
        # self (object). A pointer to the sample in use
        # param (String). String of the parameter defined in the material class to be read out

        # Returns:
        # params (numpy array). 1d-array of the parameters asked for
        # Special case for cp_T: Returns the grid on which the Einstein heat capacity is pre-computed
        # and the respective heat capacities
        # Special case for ms, s_up(dn)_eig_squared: Returns only the parameters for magnetic materials
        # Special case for kappae(p)_sam): Returns 2d-array of diffusion constants to the left (closer to the laser)
        # in kappa_e(p)_sam[:,0] and to the right in kappa_e(p)_sam[:,1], respecting the interface coefficients given in
        # self.add_layers

        if param == 'cp_T':
            return [mat.cp_T_grid for mat in self.mats], [mat.cp_T for mat in self.mats]
        elif param == 'ms':
            return np.array([mat.ms for mat in self.mat_arr if mat.muat != 0])
        elif param == 's_up_eig_squared':
            return np.array([mat.s_up_eig_squared for mat in self.mat_arr if mat.muat != 0])
        elif param == 's_dn_eig_squared':
            return np.array([mat.s_dn_eig_squared for mat in self.mat_arr if mat.muat != 0])
        elif param == 'kappae':
            kappa_e_sam = np.zeros((self.len, 2))
            kappa_e_sam[:, 1] = np.array([mat.kappae for mat in self.mat_arr])
            pos = 0
            for i, num in enumerate(self.mat_blocks):
                pos += num
                if i < len(self.mat_blocks)-1:
                    kappa_e_sam[pos-1, 1] = self.kappa_e_int[i]
            kappa_e_sam[-1, 1] = 0.
            kappa_e_sam[:, 0] = np.roll(kappa_e_sam[:, 1], shift=1, axis=0)
            return kappa_e_sam
        elif param == 'kappap':
            kappa_p_sam = np.zeros((self.len, 2))
            kappa_p_sam[:, 1] = np.array([mat.kappap for mat in self.mat_arr])
            pos = 0
            for i, num in enumerate(self.mat_blocks):
                pos += num
                if i < len(self.mat_blocks)-1:
                    kappa_p_sam[pos-1, 1] = self.kappa_p_int[i]
            kappa_p_sam[-1, 1] = 0.
            kappa_p_sam[:, 0] = np.roll(kappa_p_sam[:, 1], shift=1, axis=0)
            return kappa_p_sam
        elif param == 'cp2_T':
            return [mat.cp2_T_grid for mat in self.mats], [mat.cp2_T for mat in self.mats]
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
        # material_blocks (list). List of lists of the blocks of layers separated by changes in materials
        # along the sample, containing the number of repeated layers in each list of list.

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

    def get_tp2_mask(self):
        # This method creates a mask for the layers where a second phonon system is to be considered and computed.

        # Input:
        # self (object). The sample in use

        # Returns:
        # tp2_mask (boolean array). 1d-array selecting the layers where dynamics of a second phonon system
        # shall be computed.

        tp2_mask = self.get_params('gpp') != 0

        return tp2_mask

    def get_el_mag_mask(self):
        # This method creates a mask for free-electron layers, filtering out non-magnetic ones. Important for handling
        # of samples with free-electron, but non-magnetic layers

        # Input:
        # self (object). The sample in use

        # Returns:
        # el_mag_mask (boolean array). 1d-array of length self.len_te, with entries True for magnetic layers, False for
        # nonmagnetic layers

        el_mag_mask = np.array([mat.muat for mat in self.mat_arr[self.el_mask]]) != 0

        return el_mag_mask

    def get_mat_positions(self):
        # This method determines all different materials in the sample and creates a nested list of their positions

        # Input:
        # self (object). The sample in use

        # Returns:
        # mats (list). List of the different materials in the sample, starting with the one closest to the laser pulse.
        # mat_ind (list). List of the positions of each layer of material in the sample, positions stored in arrays

        mats = []
        for i, mat in enumerate(list(self.mat_arr)):
            if mat not in mats:
                mats.append(mat)

        mat_indices = [[] for _ in mats]
        for j, mat in enumerate(mats):
            for i in range(self.get_len()):
                if self.mat_arr[i] == mat:
                    mat_indices[j].append(i)

        return mats, [np.array(ind_list) for ind_list in mat_indices]

    def get_mat_tp2_positions(self):
        # This method determines the positions of the tp2 entries of each material with two phononic subsystems
        # in the array tp2 = tp[SimDynamics.SimSample.len(sam):]

        # Input:
        # self (Reference). Reference to sample object in use

        # Returns
        # mat_tp2_indices (list). List of numpy arrays containing the indices of materials with two phononic subsystems
        # in the tp2 array in the main simulation loop

        mat_tp2s = [mat for mat in self.mats if mat.gpp != 0]
        mat_tp2_indices = []
        start_pos = 0
        for i, mat in enumerate(mat_tp2s):
            n_mat = 0
            for all_mat in self.mat_arr:
                if all_mat == mat:
                    n_mat += 1
            mat_tp2_indices.append(np.arange(start_pos, start_pos+n_mat))
            start_pos += mat_tp2_indices[i][-1]

        return mat_tp2_indices

    def get_num_mag_mat(self):
        # This method merely counts the number of magnetic materials in the sample, determined by whether the atomic
        # magnetic moment is larger than zero.

        # Input:
        # self (object). The sample in use

        # Returns:
        # mag_counter (int). Number of magnetic layers in the sample

        mag_counter = 0
        for mat in self.mat_arr:
            if mat.muat != 0:
                mag_counter += 1

        return mag_counter

    def show_info(self):
        print()
        print('Sample constructed.')
        print('Constitiuents: ', str(self.constituents))
        print('Thicknesses:' , str(self.mat_blocks*self.dz_arr*1e9) , ' nm')
        print('Number of layers:' , str(self.mat_blocks))
        print()
