import numpy as np
from scipy import constants as sp
from matplotlib import pyplot as plt


class SimMaterials:
    # The material class holds mostly parameters of the materials in the sample to be constructed.
    # Also, it holds information like the thickness of the layers of material (dz) and penetration depth (pen_dep)

    def __init__(self, name, tdeb, cp_max, cp_method, kappap,
                 ce_gamma=0., kappae=0., gep=0.,
                 vat=0., spin=0., tc=0., muat=0., asf=0.):
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
        # is computed with Einstein model or Debye model depending on the parameter:
        # cp_method (String). 'Einstein' for Einstein model, 'Debye' for Debye model, any other string must be a file
        # location that can be read out with np.loadtxt()

        # Also returns:
        # tein (float). The approximate Einstein temperature in relation to the Debye temperature.
        # cp_T_grid, cp_T (numpy arrays). Arrays of (i) the temperature grid on which the Einstein
        # heat capacity is computed and (ii) the corresponding cp values
        # gpp. Phonon-Phonon coupling parameter. Default in None.
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
        self.cp_method = cp_method

        self.tein = 0.75*tdeb
        self.cp_T_grid, self.cp_T = self.create_cp_T(self.cp_method, self.cp_max)
        self.gpp = 0
        self.cp2_T_grid, self.cp2_T = None, None
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

    def create_cp_T(self, cp_method, cp_max):
        # This method constructs a temperature grid (fine-grained until tdeb, course-grained until 3*tdeb).
        # It computes the Einstein or Debye lattice heat capacity on this temperature grid.
        # Later we can use this grid to read out the proper heat capacity at every lattice temperature

        # Input:
        # self (object). A pointer to the material in use

        # Returns:
        # t_grid (numpy array). 1d-array of temperature grid between 0 and 3*tdeb+1 K or given range from file
        # cp_t_grid (numpy array). 1d-array of Einstein/Debye/input lattice capacity for the above temperature grid

        if cp_method == 'Einstein':
            t_grid_fine = np.arange(1, self.tdeb, 0.1)
            t_grid_course = np.arange(self.tdeb + 1, 3 * self.tdeb)
            t_grid = np.append(t_grid_fine, t_grid_course)
            t_red = np.divide(self.tein, t_grid)
            cp_t_grid = cp_max*((t_red**2*np.divide(np.exp(t_red), (np.exp(t_red)-1)**2))+0.05)

            t_grid = np.append(t_grid, 3*self.tdeb+1)
            cp_t_grid = np.append(cp_t_grid, self.cp_max)
            return t_grid, list(cp_t_grid)

        elif cp_method == 'Debye':
            t_grid = np.linspace(1, 4 * self.tdeb, 4000)

            cp_t_grid = self.debye_heat_capacity(cp_max, t_grid)
            return t_grid, list(cp_t_grid)

        else:
            file = np.loadtxt(cp_method)
            t_grid = file[:, 0]
            cp_t_grid = file[:, 1]
            return t_grid, list(cp_t_grid)

    @staticmethod
    def debye_integral(t_red, num_points):
        # This method calculates the Debye integral for a given value of reduced temperature. It is used to compute
        # the phononic heat capacity for a given temperature, as defined in the method debye_heat_capcity.

        # Input:
        # t_red (float). Reduced temperature, the upper integration limit
        # num_points (int). Number of points to define the integration grid.

        # Returns:
        # integral_approx (float). Approximated Debye integral

        if t_red == 0:
            return 0

        x = np.linspace(1e-5, t_red, num_points)
        integrand = x ** 3 / (np.exp(x) - 1)
        integral_approx = np.trapz(integrand, x)

        return integral_approx

    def debye_heat_capacity(self, cp_max, T, num_points=1000):
        # This method computes the Debye model phononic heat capacity for a given temperature grid
        # defined in create_cp_T().

        # Input:
        # self (class object). The material in use
        # T (numpy array). 1d array of a given temerpature grid
        # num_points (int). Number of points for the computation of the Debye integral in debye_integral().
        # Default is set to 1000.

        # Returns:
        # heat_capacities (numpy array). 1d array of the heat capacities, respective to the input of T

        t_red = self.tdeb / T

        heat_capacities = np.zeros_like(T)

        for i, yi in enumerate(t_red):
            D_yi = 3 * SimMaterials.debye_integral(yi, num_points) / yi ** 3 if yi != 0 else 0
            heat_capacities[i] = cp_max * D_yi

        return heat_capacities

    def add_phonon_subsystem(self, gpp, cp2_method, cp2_max):
        # This method adds parameters of phonon-phonon coupling and heat capacity of a second phononic heat capacity
        # to the initiated material.

        # Input:
        # g_pp (float). Phonon-phonon coupling parameter in W/m^3/K
        # cp2_method (String). Method to compute the second heat capacity with
        # cp2_max [optional] (float). maximum heat capacity of the second phononic subsystem in J/m^3/K

        # Returns:
        # __void function__

        self.gpp = gpp
        self. cp2_T_grid, self.cp2_T = self.create_cp_T(cp2_method, cp2_max)

        return

    def visualize_cp(self):
        # This method plots the phononic heat capacity computed with the chosen model.

        # Input:
        # self (class object). The material in use

        # Returns:
        # __void function__

        plt.figure(figsize=(8, 6))
        plt.xlabel(r'Temperature [K]', fontsize=16)
        plt.ylabel(r'$C_p$(T) [MJ/(m$^3$K)]', fontsize=16)
        plt.plot(self.cp_T_grid, np.array(self.cp_T)*1e-6, lw=2.0, label=r'$C_{p}$')
        if self.cp2_T_grid is not None and self.cp2_T is not None:
            plt.plot(self.cp2_T_grid, np.array(self.cp2_T)*1e-6, lw=2.0, label = r'$C_{p2}$')
        plt.show()

        return
