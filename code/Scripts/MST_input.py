# All changes to redefine parameters of the simulation can be done in this file.
# For documentation of the simulation methods or class parameters, see the respective files.
# Unless explicitly stated otherwise, all parameters are to be put in SI units.
# Short documentation of the simulation setup is given before each block here.

import numpy as np

# Import classes from other files to set up materials, sample, pulse and the dynamical functions:
from code.Source.mats import SimMaterials
from code.Source.sample import SimSample
from code.Source.pulse import SimPulse
from code.Source.mainsim import SimDynamics

MST = SimMaterials(name='Mn3Si2Te6', cp_max=1.47e6, cp_method='Debye', tdeb=155.,  kappap=0.,
                   ce_gamma=619, gep=4.7e17)

# MST.add_phonon_subsystem(gpp=2.5e17, cp2_max=None, cp2_method='input_data/FGT/FGT_c_p2.txt')

# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
sample.add_layers(material=MST, layers=1,  dz=1.7e-9, pen_dep=1e-9)

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, method='LB', pulse_width=25.5e-15, fluence=9.8e-3, delay=1e-12, therm_time=100e-15)
# pulse.visualize(axis='t')

# Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=6e-12, ini_temp=100., solver='RK45', max_step=1e-13)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# te should be something like solution.y.T[:, 0] -> normalize -> fit to exp data
# delay should be something like solution.t

# Save the data in a file with the desired name
sim.save_data(solution, save_file=f'MST_test')
