# All changes to redefine parameters of the simulation can be done in this file.
# For documentation of the simulation methods or class parameters, see the respective files.
# Unless explicitly stated otherwise, all parameters are to be put in SI units.
# Short documentation of the simulation setup is given before each block here.

# Import classes from other files to set up materials, sample and pulse:
from ..Source.mats import SimMaterials
from ..Source.sample import SimSample
from ..Source.pulse import SimPulse
from ..Source.mainsim import SimDynamics

cgt = SimMaterials(name='CGT', pen_dep=30e-9, tdeb=200, dz=2e-9, vat=1e-28, ce_gamma=737., cp_max=1.4e6,
                   kappap=1., kappae=0.0013, gep=15e16, spin=1.5, tc=65., muat=4., asf=0.13)
fgt = SimMaterials(name='FGT', pen_dep=30e-9, tdeb=190, dz=2e-9, vat=1.7e-29, ce_gamma=1561., cp_max=2e6,
                   kappap=0.5, kappae=0.25, gep=0.8e18, spin=1.5, tc=220., muat=2., asf=0.08)
cri3 = SimMaterials(name='CrI3', pen_dep=30e-9, tdeb=134, dz=2e-9, vat=1.35e-28, ce_gamma=550., cp_max=1.23e6,
                   kappap=1.36, kappae=0., gep=4e16, spin=1.5, tc=61., muat=4, asf=0.175)
hbn = SimMaterials(name='hBN', pen_dep=1, tdeb=400, dz=2e-9, vat=1e-28, ce_gamma=0., cp_max=2.6e6, kappap=5.0,
                   kappae=0., gep=0., spin=0., tc=0., muat=0., asf=0.)
sio2 = SimMaterials(name='SiO2', pen_dep=1, tdeb=470, dz=2e-9, vat=1e-28, ce_gamma=0., cp_max=1.7e6, kappap=1.,
                    kappae=0., gep=0., spin=0, tc=0., muat=0., asf=0.)
ain = SimMaterials(name='AIN', pen_dep=1, tdeb=1150, dz=2e-9, vat=1e-28, ce_gamma=0, cp_max=3.5e6, kappap=8.5,
                   kappae=0., gep=0., spin=0, tc=0., muat=0., asf=0.)
ws2 = SimMaterials(name='WS2', pen_dep=1, tdeb=213, dz=2e-9, vat=1e-28, ce_gamma=0, cp_max=2e6, kappap=1.7,
                   kappae=0., gep=0., spin=0, tc=0., muat=0., asf=0.)
wse2 = SimMaterials(name='WSe2', pen_dep=1, tdeb=160, dz=2e-9, vat=1e-28, ce_gamma=0, cp_max=2e6, kappap=0.35,
                   kappae=0., gep=0., spin=0, tc=0., muat=0., asf=0.)
graphene = SimMaterials(name='graphene', pen_dep=1, tdeb=1911, dz=2e-9, vat=1e-28, ce_gamma=0, cp_max=4.68e6, kappap=6.,
                   kappae=0., gep=0., spin=0, tc=0., muat=0., asf=0.)
bite = SimMaterials(name='Bi2Te3', pen_dep=1, tdeb=165, dz=2e-9, vat=1e-28, ce_gamma=0, cp_max=1.2e6, kappap=1.8,
                   kappae=0., gep=0., spin=0, tc=0., muat=0., asf=0.)
al2o3 = SimMaterials(name='Al2O3', pen_dep=1, tdeb=980, dz=2e-9, vat=1e-28, ce_gamma=0, cp_max=3.55e6, kappap=1.,
                   kappae=0., gep=0., spin=0, tc=0., muat=0., asf=0.)
mos2 = SimMaterials(name='MoS2', pen_dep=1, tdeb=280, dz=2e-9, vat=1e-28, ce_gamma=0, cp_max=2e6, kappap=5.,
                   kappae=0., gep=0., spin=0, tc=0., muat=0., asf=0.)


sub_list = [graphene, bite, al2o3, mos2]
ferro_list = [cgt, fgt, cri3]
fluence_list = [0.3, 0.5, 1., 1.5]

for sub in sub_list:
    for ferro in ferro_list:
        for flu in fluence_list:

            # Create a sample, then add desired layers of the materials you want to simulate.
            # The first material to be added will be closest to the laser pulse and so on.
            sample = SimSample()
            sample.add_layers(material=sub, layers=150)
            sample.add_layers(material=ferro, layers=8, kappap_int='av')

            # Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
            pulse = SimPulse(sample=sample, pulse_width=60e-15, fluence=flu, delay=1e-12)

            if ferro == fgt:
                T0 = 6.
            else:
                T0 = 6.

            # Initialize the simulation with starting temperature and final time, then run the solve function:
            sim = SimDynamics(sample=sample, pulse=pulse, end_time=3e-9, ini_temp=T0, solver='RK45', max_step=1e-13)

            # Run the simulation by calling the function that creates the map of all three baths
            solution = sim.get_t_m_maps()

            # Save the data in a file with the desired name
            sim.save_data(solution, save_file='array_cap/' + str(ferro.name) + '(16)_' + str(sub.name) + '(300)_' + 'flu_' + str(flu))
