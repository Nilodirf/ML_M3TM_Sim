# All changes to redefine parameters of the simulation can be done in this file.
# For documentation of the simulation methods or class parameters, see the respective source code files.
# Unless explicitly stated otherwise, all parameters are to be put in SI units.
# Short documentation of the simulation setup is given before each block here.

# Import classes from other files to set up materials, sample, pulse and main simulation:
from mats import SimMaterials
from sample import SimSample
from pulse import SimPulse
from mainsim import SimDynamics


# Create the necessary materials. For documentation of the parameters see mats.SimMaterials class:

dummy = SimMaterials(name='dummy', dz=2e-9,
                     cp_max=[2.645e6, 2.5e6], tdeb=[400., 700.], kappap=[5., 3.],
                     ce_gamma=[100., 10., 0.5], kappae=[10., 100., 0.], gep=[[1e15, 1e15], [2e15, 2e15], [1e15, 1.5e15]],
                     spin=[2., 3., 0.], tc=100., muat=[1., 2., 0.], asf=[0.07, 0.05, 0.], vat=[1e-28, 2e-28, 1])
dummy_2 = SimMaterials(name='dummy_2', dz=2e-9,
                     cp_max=[2.645e6], tdeb=[400.], kappap=[5.],
                     ce_gamma=[100.], kappae=[10.], gep=[[1e15]],
                     spin=[2.], tc=100., muat=[1.], asf=[0.07], vat=[1e-28])
# CGT = SimMaterials(name='CGT', tdeb=200, dz=2e-9, vat=1e-28, ce_gamma=737., cp_max=1.4e6, kappap=1., kappae=0., gep=15e16, spin=1.5, tc=65., muat=2., asf=0.05)
# SiO2 = SimMaterials(name='SiO2', tdeb=403, dz=2e-9, vat=1e-30, ce_gamma=0., cp_max=1.9e6, kappap=1.5, kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)
dummy.visualize_cp()
# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
sample.add_layers(material=dummy, layers=5, n_comp=1+3j, pen_dep=1)
sample.add_layers(material=dummy_2, layers=6, kappap_int='av', kappae_int='av', n_comp=1.5+2j)
# sample.add_layers(material=CGT, layers=100, kappap_int='av', n_comp=1.5+5j, pen_dep=10e-9)
# sample.add_layers(material=SiO2, layers=5, kappap_int='av', n_comp=1.5+0j, pen_dep=1)

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, pulse_width=20e-15, fluence=0.5, delay=1e-12, pulse_dt=1e-16, method='Abeles',
                 theta=0, phi=1/2, energy=1)
pulse.visualize(axis='z')

# # Initialize the simulation with starting temperature and final time, then run the solve function:
# sim = SimDynamics(sample=sample, pulse=pulse, end_time=10e-12, ini_temp=30., constant_cp=False, ep_eq_dt=1e-16,
#                   long_time_dt=1e-14, solver='RK45', max_step=1e-12)
#
# # Run the simulation by calling the function that creates the map of all three baths
# solution = sim.get_t_m_maps()
#
# # Save the data in a file with the desired name
# sim.save_data(solution, save_file='14_nm')
