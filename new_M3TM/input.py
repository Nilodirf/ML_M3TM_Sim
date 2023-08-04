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

mgo = SimMaterials(name='MgO', pen_dep=1, tdeb=400, dz=0.2e-9, vat=1e-28, ce_gamma=0., cp_max=4e6, kappap=42.,
                   kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)
iron = SimMaterials(name='Fe', pen_dep=17e-9, tdeb=300, dz=0.2e-9, vat=1e-29, ce_gamma=750., cp_max=3.5e6,
                   kappap=8., kappae=10., gep=2e18, spin=2, tc=1024., muat=2., asf=0.02)

# mgo = SimMaterials(name='MgO', pen_dep=1, tdeb=400, dz=0.4e-9, vat=1e-28, ce_gamma=0., cp_max=4e6, kappap=42.,
#                    kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)
# iron = SimMaterials(name='Fe', pen_dep=17e-9, tdeb=300, dz=0.2e-9, vat=1e-30, ce_gamma=750., cp_max=3.5e6,
#                    kappap=8., kappae=80., gep=2e18, spin=2, tc=1024., muat=2., asf=0.01)

# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
sample.add_layers(material=mgo, layers=10)
sample.add_layers(material=iron, layers=10, kappap_int='av')
sample.add_layers(material=mgo, layers=100, kappap_int='av')

# sample = SimSample()
# sample.add_layers(material=mgo, layers=10)
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')
# sample.add_layers(material=iron, layers=10, kappap_int='av')
# sample.add_layers(material=mgo, layers=10, kappap_int='av')

# sample = SimSample()
# sample.add_layers(material=mgo, layers=5)
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')
# sample.add_layers(material=iron, layers=5, kappap_int='av')
# sample.add_layers(material=mgo, layers=5, kappap_int='av')


# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, pulse_width=20e-15, fluence=6., delay=1e-12)

# Initialize the simulation with starting temperature and final time, then run the solve function:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=10e-12, ini_temp=295., constant_cp=False)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# Save the data in a file with the desired name
sim.save_data(solution, save_file='mgo_fe_mgo')
