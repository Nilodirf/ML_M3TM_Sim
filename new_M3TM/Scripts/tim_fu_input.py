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
Yttrium = SimMaterials(name='Y', tdeb=186, vat=1e-28, ce_gamma=400., cp_max=1.25e6, kappap=7.,
                   kappae=10., gep=0.5e17, spin=0., tc=0., muat=0., asf=0.)
Terbium = SimMaterials(name='Tb', tdeb=174, vat=1e-28, ce_gamma=225., cp_max=2.2e6,
                   kappap=7., kappae=10., gep=2.5e17, spin=0., tc=0., muat=0., asf=0.)
Aluminium = SimMaterials(name='Al', tdeb=390, vat=1e-28, ce_gamma=135., cp_max=2.5e6, kappap=0.,
                    kappae=240., gep=3e17, spin=0, tc=0., muat=0., asf=0.)
Siliconnitirde = SimMaterials(name='Si3N4', tdeb=400, vat=1.7e-29, ce_gamma=0., cp_max=3.17e6,
                   kappap=30., kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)


# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
# sample.add_layers(material=Yttrium, dz=1e-9, layers=2, pen_dep=34e-9, n_comp=2.21+2.73j)
# sample.add_layers(material=Terbium, dz=1e-9, layers=10, pen_dep=19.4e-9, n_comp=1.97+3.28j, kappap_int='av', kappae_int='max')
# sample.add_layers(material=Yttrium, dz=1e-9, layers=20, pen_dep=34e-9, n_comp=2.21+2.73j, kappap_int='av', kappae_int='max')
# sample.add_layers(material=Aluminium, dz=1e-9, layers=300, pen_dep=7.5e-9, n_comp=2.8+8.45j, kappap_int='av', kappae_int='max')
# sample.add_layers(material=Siliconnitirde, dz=1e-9, layers=200, pen_dep=1, n_comp=2.008+0j, kappap_int='av')

# sample.add_layers(material=Yttrium, dz=2e-9, layers=1, pen_dep=34e-9, n_comp=2.21+2.73j)
# sample.add_layers(material=Terbium, dz=2e-9, layers=5, pen_dep=19.4e-9, n_comp=1.97+3.28j, kappap_int='av', kappae_int='max')
# sample.add_layers(material=Yttrium, dz=2e-9, layers=10, pen_dep=34e-9, n_comp=2.21+2.73j, kappap_int='av', kappae_int='max')
# sample.add_layers(material=Aluminium, dz=2e-9, layers=150, pen_dep=7.5e-9, n_comp=2.8+8.45j, kappap_int='av', kappae_int='max')
# sample.add_layers(material=Siliconnitirde, dz=2e-9, layers=100, pen_dep=1, n_comp=2.008+0j, kappap_int='av')

sample.add_layers(material=Terbium, dz=0.01e-9, layers=550, pen_dep=19.4e-9, n_comp=1.97+3.28j)


# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, method='Abeles', pulse_width=20e-15, fluence=5., delay=0.5e-12, photon_energy_eV=1.55, theta=0, phi=0)
pulse.visualize(axis='z', save_fig=True, save_file='tim_only_Tb_absorption_Abeles', fit=None)

# Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=5e-12, ini_temp=300., solver='RK23', max_step=1e-16)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# Save the data in a file with the desired name
sim.save_data(solution, save_file='tim_only_Tb_Abeles')