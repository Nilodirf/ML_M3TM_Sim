import numpy as np
from scipy import constants as sp


class SimMaterials:
    # The material class holds mostly parameters of the materials in the sample to be constructed.
    # Also, it holds information like the thickness of the layers of material (dz) and penetration depth (pen_dep)

    def __init__(self, name, pen_dep, tdeb, dz, vat, ce_gamma, cp_max, kappap, kappae, gep, spin, tc, muat, asf):
        self.name = name
        self.pen_dep = pen_dep
        self.tdeb = tdeb
        self.dz = dz
        self.vat = vat
        self.kappap = kappap
        self.kappae = kappae
        self.gep = gep
        self.spin = spin
        self.tc = tc
        self.muat = muat
        self.asf = asf
        self.ce_gamma = ce_gamma
        self.cp_max = cp_max
        self.tein = 0.75*tdeb
        self.cp_T_grid, self.cp_T = self.create_cp_T()
        if muat == 0:
            self.R = 0
            self.J = 0
            self.arbsc = 0
            self.ms = None
            self.s_up_eig_squared = None
            self.s_dn_eig_squared = None
        else:
            self.R = 8 * self.asf * self.vat * self.tc**2 / self.tdeb**2 / sp.k / self.muat
            self.J = 3 * sp.k * self.tc * self.spin / (self.spin+ 1 )
            self.arbsc = self.R / self.tc**2 / sp.k * self.gep
            self.ms = (np.arange(2 * self.spin + 1) + np.array([-self.spin for i in range(int(2 * self.spin) + 1)]))
            self.s_up_eig_squared = -np.power(self.ms, 2) - self.ms + self.spin ** 2 + self.spin
            self.s_dn_eig_squared = -np.power(self.ms, 2) + self.ms + self.spin ** 2 + self.spin

    def create_cp_T(self):
        # This method constructs a temperature grid (fine-grained until tdeb, course-grained until 3*tdeb).
        # It computes the Einstein lattice heat capacity on this temperature grid.
        # Later we can use this grid to read out the proper heat capacity at every lattice temperature

        # Input:
        # self (object). A pointer to the material in use

        # Returns:
        # t_grid (numpy array). 1d-array of temperature grid between 0 and 3*tdeb+1 K
        # cp_t_grid (numpy array). 1d-array of Einstein lattice capacity for the above temperature grid,
        # the last value being cp_max for every temperature above 3*tdeb+1 K

        t_grid_fine = np.arange(1, self.tein, 0.1)
        t_grid_course = np.arange(self.tein+1, 3*self.tein)
        t_grid = np.append(t_grid_fine, t_grid_course)
        t_red = np.divide(self.tein, t_grid)
        cp_t_grid = self.cp_max*t_red**2*np.divide(np.exp(t_red), (np.exp(t_red)-1)**2)

        t_grid = np.append(t_grid, 3*self.tdeb+1)
        cp_t_grid = np.append(cp_t_grid, self.cp_max)
        return t_grid, cp_t_grid
