import numpy as np
from scipy import constants as sp


class SimMaterials:
    # The material class holds mostly parameters of the materials in the sample to be constructed.
    # Also, it holds information like the thickness of the layers of material (dz) and penetration depth (pen_dep)

    def __init__(self, name, tdeb, vat, ce_gamma, cp_max, kappap, kappae, gep, spin, tc, muat, asf):
        # Input:
        # name (String). Name of the material
        # tdeb (float). Debye temperature of the material
        # vat (float). Magnetic atomic volume in m. Influences for magnetization rate parameter in M3TM
        # kappap (float). Phononic heat diffusion constant in W/m/K
        # kappae (float). Electronic heat diffusion constant in W/m/K (set to 0 if no itinerant electrons)
        # gep (float). Electron-phonon coupling constant in W/m**3/K (set to 0 if no itinerant electrons)
        # spin (float). Effective spin of the material (set to 0 if no itinerant spin-full electrons)
        # tc (float). Curie temperature of the material (set to 0 if not magnetic)
        # muat (float). Atomic magnetic moment in unit of mu_bohr. (set to 0 if not magnetic)
        # asf (float). Electron-phonon-scattering induced spin flip probability
        # of the material (set to 0 if no itinerant spin-full electrons)
        # ce_gamma (float). Sommerfeld constant of electronic heat capacity
        # in J/m**3/K (set to 0 if no itinerant electrons)
        # cp_max (float). Maximal phononic heat capacity in W/m**3/K. Temperature dependence
        # is computed with Einstein model

        # Also returns:
        # tein (float). The approximate Einstein temperature in relation to the Debye temperature.
        # cp_T_grid, cp_T (numpy arrays). Arrays of (i) the temperature grid on which the Einstein
        # heat capacity is computed and (ii) the corresponding cp values
        # R (float/None). magnetization rate parameter in 1/s, None if muat == 0
        # J (float/None). Mean field exchange coupling parameter, None if muat == 0
        # arbsc (float/None). Closely related to rate parameter, None if muat == 0
        # ms (numpy array/None). Spin-levels depending on the effective spin, None if muat == 0
        # s_up_eig_squared (numpy array/None). Eigenvalues of the creation spin ladder operator for all spin sublevels
        # s_dn_eig_squared (numpy array/None). Eigenvalues of the annihilation spin ladder operator
        # for all spin sublevels

        self.name = name
        self.tdeb = tdeb
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
            self.R = 8 * self.asf * self.vat * self.tc**2 / self.tdeb**2 / sp.k / self.muat * self.gep
            self.J = 3 * sp.k * self.tc * self.spin / (self.spin + 1)
            self.arbsc = self.R / self.tc**2 / sp.k
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
        # the last value being cp_max for every temperature above 3*tein+1 K

        t_grid_fine = np.arange(1, self.tein, 0.1)
        t_grid_course = np.arange(self.tein+1, 3*self.tein)
        t_grid = np.append(t_grid_fine, t_grid_course)
        t_red = np.divide(self.tein, t_grid)
        cp_t_grid = self.cp_max*((t_red**2*np.divide(np.exp(t_red), (np.exp(t_red)-1)**2))+0.1)

        t_grid = np.append(t_grid, 3*self.tein+1)
        cp_t_grid = np.append(cp_t_grid, self.cp_max)
        return t_grid, list(cp_t_grid)
