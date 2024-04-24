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
Yttrium = SimMaterials(name='Y', pen_dep=34e-9, tdeb=186, dz=1e-9, vat=1e-28, ce_gamma=400., cp_max=1.25e6, kappap=7.,
                   kappae=10., gep=0.5e17, spin=0., tc=0., muat=0., asf=0.)
Terbium = SimMaterials(name='Tb', pen_dep=19.4e-9, tdeb=174, dz=1e-9, vat=1e-28, ce_gamma=225., cp_max=2.2e6,
                   kappap=35., kappae=60., gep=2.5e17, spin=0., tc=0., muat=0., asf=0.)
Aluminium = SimMaterials(name='Al', pen_dep=7.5e-9, tdeb=390, dz=1e-9, vat=1e-28, ce_gamma=135., cp_max=2.5e6, kappap=40.,
                    kappae=200., gep=3e17, spin=0, tc=0., muat=0., asf=0.)
Siliconnitirde = SimMaterials(name='Si3N4', pen_dep=1, tdeb=400, dz=1e-9, vat=1.7e-29, ce_gamma=0., cp_max=3.17e6,
                   kappap=30., kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)


# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
sample.add_layers(material=Yttrium, layers=2)
sample.add_layers(material=Terbium, layers=10, kappap_int='av', kappae_int='max')
sample.add_layers(material=Yttrium, layers=20, kappap_int='av', kappae_int='max')
sample.add_layers(material=Aluminium, layers=300, kappap_int='av', kappae_int='max')
sample.add_layers(material=Siliconnitirde, layers=200, kappap_int='av')

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, pulse_width=20e-15, fluence=0.5, delay=1e-12)

# Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=5e-12, ini_temp=300., solver='RK45', max_step=1e-16)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# Save the data in a file with the desired name
sim.save_data(solution, save_file='tim_full_sam_LB')