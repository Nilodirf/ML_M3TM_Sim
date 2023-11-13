import numpy as np
from scipy import constants as sp


class SimMaterials:
    # The material class holds mostly parameters of the materials in the sample to be constructed.
    # Also, it holds information like the thickness of the layers of material (dz) and penetration depth (pen_dep)

    def __init__(self, name, tdeb, dz, cp_max, kappap, ce_gamma, kappae, gep, spin, tc, asf, vat, muat=0):
        # Input:
        # name (String). Name of the material
        # tdeb (list). List of Debye temperatures in K of all phononic subsystems of the material.
        # dz (float). Layer thickness of the material in m. Important only for resolution of heat diffusion
        # and pump pulse.
        # vat (float). Magnetic atomic volumes in m**3.
        # kappap (list). List of phononic heat diffusion constants in W/m/K for all phononic subsystems in the material.
        # kappae (list). List of electronic heat diffusion constants in W/m/K for all electronic subsystems
        # of the material.
        # gep (list of lists). Matrix Electron(i)-phonon(j) coupling constants g(ij) in W/m**3/K for interactions
        # between all electronic and phononic subsystems.
        # spin (list). List of effective spins of all magnetic subsystems in the material.
        # tc (float). Curie temperature of the material (set to 0 if not magnetic)
        # muat (list). List of atomic magnetic moments in units of mu_bohr for all magnetic subsystems in the material.
        # asf (list). List of electron-phonon-scattering induced spin flip probabilities for all magnetic subsystems
        # of the material.
        # ce_gamma (list). List of Sommerfeld constants of electronic heat capacity of all electronic subsystems
        # in J/m**3/K (set to 0 if no itinerant electrons).
        # cp_max (list). List of maximum phononic heat capacities in W/m**3/K for all phononic subsystems
        # of the material.

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
        self.tdeb = np.array(tdeb)
        self.dz = dz
        self.vat = vat
        self.kappap = np.array(kappap)
        self.kappae = np.array(kappae)
        self.gep = np.array(gep)
        self.spin = np.array(spin)
        self.tc = tc
        self.muat = np.array(muat)
        self.asf = np.array(asf)
        self.ce_gamma = np.array(ce_gamma)
        self.cp_max = np.array(cp_max)
        self.tein = 0.75*self.tdeb
        self.cp_T_grid, self.cp_T = self.create_cp_T()
        if muat == 0:
            self.R = None
            self.J = None
            self.arbsc = None
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
        # This method constructs a temperature grid, and the Einstein lattice heat capacity on this temperature grid.
        # Later we can use this grid to read out the proper heat capacity at any lattice temperature

        # Input:
        # self (object). A pointer to the material in use

        # Returns:
        # t_grid (numpy array). 1d-array of temperature grid between 0 and 3*tdeb+1 K
        # cp_t_grid (numpy array). 1d-array of Einstein lattice capacity for the above temperature grid,
        # the last value being cp_max for every temperature above 3*tdeb+1 K

        t_grid = np.arange(1., 2*self.tein, 0.1)
        t_red = np.divide(self.tein, t_grid)
        cp_t_grid = self.cp_max*(t_red**2*np.divide(np.exp(t_red), (np.exp(t_red)-1)**2))

        t_grid = np.append(t_grid, 2*self.tein+1)
        cp_t_grid = np.append(cp_t_grid, self.cp_max)
        return t_grid, list(cp_t_grid)
