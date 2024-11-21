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

gpps = np.arange(10, 41)*1e16
# Fluence with pen_dep=1nm; Te:9.8e-3, mag:49e-3, Tp:38e-3
# Starting temperatures; Te:100K, mag:25K, Tp:100K

fluences = [9.8e-3, 38e-3]
initemps = [100., 100.]
subsystems = ['el', 'ph']

for g in gpps:
    for flu, it, subsys in zip(fluences, initemps, subsystems):

        FGT = SimMaterials(name='Fe3GeTe2', cp_max=None, cp_method='input_data/FGT/FGT_c_p1.txt', tdeb=232.,  kappap=0.,
                           ce_gamma=211., gep=5.5*1e17,
                           asf=0.018, spin=2, tc=232., vat=127.76e-30, muat=1.6)

        FGT.add_phonon_subsystem(gpp=g, cp2_max=None, cp2_method='input_data/FGT/FGT_c_p2.txt')

        # Create a sample, then add desired layers of the materials you want to simulate.
        # The first material to be added will be closest to the laser pulse and so on.
        sample = SimSample()
        sample.add_layers(material=FGT, layers=1,  dz=1.7e-9, pen_dep=1e-9)

        # Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
        pulse = SimPulse(sample=sample, method='LB', pulse_width=15e-15, fluence=flu, delay=1e-12, therm_time=13e-15)
        # pulse.visualize(axis='t')

        # Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
        sim = SimDynamics(sample=sample, pulse=pulse, end_time=6e-12, ini_temp=it, solver='RK45', max_step=1e-13)

        # Run the simulation by calling the function that creates the map of all three baths
        solution = sim.get_t_m_maps()


        # Save the data in a file with the desired name
        sim.save_data(solution, save_file='FGT/fits_fin/' + str(subsys) + 'g' + str(g))
