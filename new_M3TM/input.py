# All changes to redefine parameters of the simulation can be done in this file.
# For documentation of the simulation methods or class parameters, see the respective files.
# Unless explicitly stated otherwise, all parameters are to be put in SI units.
# Short documentation of the simulation setup is given before each block here.

# Import classes from other files to set up materials, sample and pulse:
from mats import SimMaterials
from sample import SimSample
from pulse import SimPulse
from mainsim import SimDynamics


# Create the necessary materials. For documentation of the parameters see mats.sim_materials class:
cgt = SimMaterials(name='CGT', pen_dep=10e-9, tdeb=175, dz=20.5e-10, vat=1e-29, ce_gamma=737., cp_max=8.9e4,
                   kappap=0.002, kappae=0.01, gep=15e16, spin=1.5, tc=65., muat=4., asf=0.3)
sio2 = SimMaterials(name='SiO2', pen_dep=1, tdeb=403, dz=5.4e-10, vat=1e-28, ce_gamma=0., cp_max=4e4, kappap=14.,
                    kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)
hbn = SimMaterials(name='hBN', pen_dep=1, tdeb=400, dz=7.7e-10, vat=1e-28, ce_gamma=0., cp_max=4e4, kappap=36.,
                   kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)

# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
sample.add_layers(material=hbn, layers=10)
sample.add_layers(material=cgt, layers=10)
sample.add_layers(material=sio2, layers=300)

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, pulse_width=20e-15, fluence=1.3, delay=1e-12)

# Initialize the simulation with starting temperature and final time, then run the solve function:
sim = SimDynamics(sample, pulse, end_time=1e-10, ini_temp=5., constant_cp=True)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# Save the data in a file with the desired name
sim.save_data(solution, save_file='try_1')
