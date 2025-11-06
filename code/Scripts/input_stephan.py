# All changes to redefine parameters of the simulation can be done in this file.
# For documentation of the simulation methods or class parameters, see the respective files.
# Unless explicitly stated otherwise, all parameters are to be put in SI units.
# Short documentation of the simulation setup is given before each block here.

# Import classes from other files to set up materials, sample, pulse and the dynamical functions:
from ..Source.mats import SimMaterials
from ..Source.sample import SimSample
from ..Source.pulse import SimPulse
from ..Source.mainsim import SimDynamics

# Create the necessary materials. For documentation of the parameters see mats.sim_materials class:
cgt = SimMaterials(name='CGT', tdeb=200, ce_gamma=737.87, cp_max=1.4e6,
                   kappap=1., kappae=0.0016, gep=15e16, cp_method='Debye')
cgt2 = SimMaterials(name='CGT', tdeb=200, ce_gamma=737.87, cp_max=1.4e6,
                   kappap=1., kappae=0.0016, gep=15e16, cp_method='Debye')
cgt3 = SimMaterials(name='CGT', tdeb=200, ce_gamma=737.87, cp_max=1.4e6,
                   kappap=1., kappae=0.0016, gep=15e16, cp_method='Debye')
sample = SimSample()
sample.add_layers(material=cgt, layers=1000,  dz=10e-9, n_comp=4.+1.8j)
sample.add_layers(material=cgt2, layers=100,  dz=100e-9, n_comp=4.+1.8j, kappap_int=10., kappae_int=0.016)
sample.add_layers(material=cgt3, layers=80,  dz=1000e-9, n_comp=1. + 0.j, kappap_int=10., kappae_int=0.016)

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, method='Abeles', pulse_width=100e-15, fluence=0.0, delay=2e-12,
                 phi=18/90, theta=0, photon_energy_ev=1.55)

# Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=1.6e-6, ini_temp=45., solver='Radau', max_step=1e-7, load_sim='first_try_stephan')

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# Save the data in a file with the desired name
sim.save_data(solution, save_file='first_try_stephan_2')
