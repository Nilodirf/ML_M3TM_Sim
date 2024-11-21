from ..Source.mats import SimMaterials
from ..Source.sample import SimSample
from ..Source.pulse import SimPulse
from ..Source.mainsim import SimDynamics

my_insulator = SimMaterials(name='Gummydummy', tdeb=400, cp_max=3e6, kappap=30.)
my_conductor = SimMaterials(name='Blitzydummy', tdeb=300, cp_max=2.5e6, kappap=20., ce_gamma=100, kappae=100., gep=1e18)
my_magnet = SimMaterials(name='Spinnydummy', tdeb=200, cp_max=2e6, kappap=10., ce_gamma=75, kappae=150., gep=0.8e18, spin=2.5, vat=1e-28, tc=600., muat=5., asf=0.06)

my_sample = SimSample()
my_sample.add_layers(material=my_conductor, dz=1e-9, layers=5, pen_dep=7.5e-9, n_comp=2.8+8.5j)
my_sample.add_layers(material=my_magnet, dz=1e-9, layers=15, pen_dep=34e-9, n_comp=2.2+2.7j, kappae_int='max', kappap_int=1.)
my_sample.add_layers(material=my_insulator, dz=1e-9, layers=300, pen_dep=1, n_comp=1.2+0j, kappap_int='av')

my_pulse_Abeles = SimPulse(sample=my_sample, method='Abeles', pulse_width=20e-15, fluence=5., delay=0.5e-12, photon_energy_ev=1.55, theta=1/4, phi=1/3)
my_pulse_Lambert_Beer = SimPulse(sample=my_sample, method='LB', pulse_width=20e-15, fluence=5., delay=0.5e-12)
my_pulse_Abeles.visualize(axis='z', save_fig=True, save_file='tutorial_pulse_Abeles')

my_simulation = SimDynamics(sample=my_sample, pulse=my_pulse_Abeles, ini_temp=300., end_time=20e-12, solver='Radau', max_step=1e-13)
my_results = my_simulation.get_t_m_maps()
my_simulation.save_data(my_results, save_file='my_result_files')
