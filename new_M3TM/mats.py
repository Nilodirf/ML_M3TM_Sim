import numpy as np

class materials:
    # The material class holds mostly parameters of the materials in the sample to be constructed.
    # Also, it holds information like the thickness of the layers of material (dz) and penetration depth (pen_dep)
    def __init__(self, name, pen_dep, tdeb, dz, apc, ce_gamma, cp_max, kappap, kappae, gep, spin, tc, muat, asf):
        self.name = name
        self.pen_dep = pen_dep
        self.tdeb = tdeb
        self.dz = dz
        self.apc = apc
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
        self.cp_T=self.create_cp_T()

    def create_cp_T(self):
        # This method constructs a temperature grid (fine-grained until tdeb, course-grained until 3*tdeb).
        # It computes the Einstein lattice heat capacity on this temperature grid.
        # Later we can use this grid to read out the proper heat capacity at every lattice temperature

        # Input:
        # self (object). A pointer to the material in use

        # Returns:
        # t_grid (numpy array). Temperature grid between 0 and 3*tdeb+1 K
        # cp_t_grid (numpy array). Einstein lattice capacity on a 1d-array for the above temperature grid,
        # the last value being cp_max for every temperature above 3*tdeb+1 K
        t_grid_fine = np.arange(0,self.tdeb, 0.1)
        t_grid_course = np.arange(self.tdeb+1, 3*self.tdeb)
        t_grid = np.append(t_grid_fine, t_grid_course)
        t_red = np.divide(self.tein/t_grid)
        cp_t_grid = self.cp_max*t_red**2*np.divide(np.exp(t_red), np.exp(t_red-1)**2)

        t_grid=np.append(t_grid, 3*self.tdeb+1)
        cp_t_grid=np.append(cp_t_grid, self.cp_max)
        return t_grid, cp_t_grid


cgt = materials(name='CGT', pen_dep=1e-9, tdeb=175, dz=20.5e-10, apc=10, ce_gamma=737., cp_max=8.9e4, kappap=0.002,
                kappae=0., gep=15e16, spin=1.5, tc=65., muat=4., asf=0.01)
sio2 = materials(name='SiO2', pen_dep=10e-9, tdeb=403, dz=5.4e-10, apc=9, ce_gamma=0., cp_max=4e6, kappap=14.,
                 kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)
hbn = materials(name='hBN', pen_dep=3e-9, tdeb=400, dz=7.7e-10, apc=4, ce_gamma=0., cp_max=4e6, kappap=36.,
                kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)
