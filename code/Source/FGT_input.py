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

FGT = SimMaterials()

fgt.visualize_cp()

# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.

sample = SimSample()
sample.add_layers(material=cgt, layers=7,  dz=2e-9, kappap_int=100., pen_dep=30e-9, n_comp=4.+1.8j)

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, method='Abeles', pulse_width=60e-15, fluence=0.5, delay=1e-12, photon_energy_ev=1.55, phi=1/2, theta=1/2*8/9)
pulse.visualize(axis='z')

# Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=1e-11, ini_temp=6., solver='RK45', max_step=1e-13)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# Save the data in a file with the desired name
sim.save_data(solution, save_file='FGT/fits')
