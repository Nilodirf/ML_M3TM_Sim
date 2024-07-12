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
hbn = SimMaterials(name='hBN', tdeb=400, vat=1e-28, ce_gamma=0., cp_max=2.645e6, kappap=5.0,
                   kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0., cp_method='Einstein')
cgt = SimMaterials(name='CGT', tdeb=200, vat=1e-28, ce_gamma=737.87, cp_max=1.4e6,
                   kappap=1., kappae=0.0016, gep=15e16, spin=1.5, tc=65., muat=4., asf=0.05, cp_method='Einstein')
sio2 = SimMaterials(name='SiO2', tdeb=470, vat=1e-28, ce_gamma=0., cp_max=2e6, kappap=1.5,
                    kappae=0., gep=0, spin=0, tc=0., muat=0., asf=0., cp_method='Einstein')
# fgt = SimMaterials(name='FGT', tdeb=190, vat=1.7e-29, ce_gamma=1561., cp_max=2e6,
#                    kappap=0.5, kappae=0.25, gep=1e18, spin=2, tc=220., muat=1.5, asf=0.06, cp_method='Debye')
# cri3 = SimMaterials(name='CrI3', tdeb=134, vat=1.35e-28, ce_gamma=550., cp_max=1.23e6,
#                    kappap=1.36, kappae=0., gep=4e16, spin=1.5, tc=61., muat=4, asf=0.175, cp_method='Debye')
# bp = SimMaterials(name='BP', tdeb=370, vat=1e-28, ce_gamma=107, cp_max=2.17e6, kappap=6.5,
#                   kappae=0.01, gep=0.9e15, spin=0, tc=0., muat=0., asf=0., cp_method='Debye')
cgt.visualize_cp()

# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
sample.add_layers(material=hbn, layers=7, dz=2e-9, pen_dep=1, n_comp=2.1+0j)
sample.add_layers(material=cgt, layers=7,  dz=2e-9, kappap_int=0.1e9, pen_dep=30e-9, n_comp=4.+1.8j)
sample.add_layers(material=sio2, layers=150, dz=2e-9, kappap_int=0.2e9, pen_dep=1, n_comp=1.45)

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, method='LB', pulse_width=60e-15, fluence=0.5, delay=1e-12, photon_energy_ev=1.55, phi=1/2, theta=1/4)
pulse.visualize(axis='z')

# Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=5e-9, ini_temp=6., solver='RK45', max_step=1e-13)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# Save the data in a file with the desired name
sim.save_data(solution, save_file='CGT Paper/TBC_100')
