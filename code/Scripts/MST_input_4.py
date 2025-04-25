# All changes to redefine parameters of the simulation can be done in this file.
# For documentation of the simulation methods or class parameters, see the respective files.
# Unless explicitly stated otherwise, all parameters are to be put in SI units.
# Short documentation of the simulation setup is given before each block here.

from scipy.optimize import differential_evolution
import numpy as np
from matplotlib import pyplot as plt

# Import classes from other files to set up materials, sample, pulse and the dynamical functions:
from code.Source.mats import SimMaterials
from code.Source.sample import SimSample
from code.Source.pulse import SimPulse
from code.Source.mainsim import SimDynamics
from code.Source.finderb import finderb

####### Define temperature and pulse for simulations:
temp_0 = 50.
fluence = 9.8e-3

####### Load exp data:
exp_data = np.loadtxt(f'input_data/MST/exp_data/new_experi_{int(temp_0)}k.dat')
exp_delay = exp_data[:, 0]
exp_te = exp_data[:, 1]

###### set initial fit values:
bounds = [(600, 1000), (0.15, 0.35), (8, 9), (0.1, 0.4)]

therm_time_initial = (bounds[0][0]+bounds[0][1])/2  # in fs
gep_initial = (bounds[1][0]+bounds[1][1])/2         # in 1e17 W/m^3/K
te_scaling_initial = (bounds[2][0]+bounds[2][1])/2  # a.u.
t0_shift_initial = (bounds[3][0]+bounds[3][1])/2    # in ps

p0 = [therm_time_initial, gep_initial, te_scaling_initial, t0_shift_initial]


####### Simulation to fit:
def fit_te_to_exp(exp_delay, therm_time_test, gep_test, te_scaling_test, t0_shift_test):

    MST = SimMaterials(name='Mn3Si2Te6', cp_max=1.28e6, cp_method='Debye', tdeb=159.,  kappap=0.,
                       ce_gamma='input_data/MST/MST_cmag.txt', gep=gep_test*1e17)

    # MST.add_phonon_subsystem(gpp=2.5e17, cp2_max=0.2e6, cp2_method='Debye')

    # Create a sample, then add desired layers of the materials you want to simulate.
    # The first material to be added will be closest to the laser pulse and so on.
    sample = SimSample()
    sample.add_layers(material=MST, layers=1,  dz=1e-9, pen_dep=1e-9)

    # Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
    pulse = SimPulse(sample=sample, method='LB', pulse_width=60e-15, fluence=fluence, delay=1.5e-12, therm_time=therm_time_test*1e-15)

    # Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
    sim = SimDynamics(sample=sample, pulse=pulse, end_time=13.6e-12, ini_temp=temp_0, solver='Radau', max_step=1e-12)

    # Run the simulation by calling the function that creates the map of all three baths
    solution = sim.get_t_m_maps()

    # extract te and use binary search to fine te at the experimental delays
    sim_delay = solution.t*1e12-1.5-t0_shift_test
    sim_te = solution.y.T[:, 0]

    sim_delay_exp = finderb(exp_delay, sim_delay)
    sim_te_exp = sim_te[sim_delay_exp]

    # normalize to zero and scale with last fit parameter:
    sim_te_exp -= sim_te[0]
    sim_te_exp /= np.amax(sim_te_exp)
    sim_te_exp *= te_scaling_test*1e-3

    print(f"THERM_TIME: {therm_time_test}\n"
          f"GEP: {gep_test*1e2}\n"
          f"TE_SCALING: {te_scaling_test}\n"
          f"T_0_SHIFT: {t0_shift_test}\n")

    return sim_te_exp


def loss(params):
    sim_te = fit_te_to_exp(exp_delay, *params)
    return np.sum((sim_te - exp_te)**2)


####### Fit:
result = differential_evolution(loss, bounds)
print(result)

therm_time_fit = result.x[0]
gep_fit = result.x[1]
te_scaling_fit = result.x[2]
t0_shift_fit = result.x[3]


###### Recreate fit
MST = SimMaterials(name='Mn3Si2Te6', cp_max=1.28e6, cp_method='Debye', tdeb=159.,  kappap=0.,
                       ce_gamma='input_data/MST/MST_cmag.txt', gep=gep_fit*1e17)

# MST.add_phonon_subsystem(gpp=2.5e17, cp2_max=0.2e6, cp2_method='Debye')

# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
sample.add_layers(material=MST, layers=1,  dz=1e-9, pen_dep=1e-9)

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, method='LB', pulse_width=25.5e-15, fluence=fluence, delay=1e-12, therm_time=therm_time_fit*1e-15)

# Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=13e-12, ini_temp=temp_0, solver='Radau', max_step=1e-12)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# sim.save_data(solution, f'MST/global_fit_{int(temp_0)}K')

# extract te and use binary search to fine te at the experimental delays
sim_delay = solution.t*1e12-1-t0_shift_fit
sim_te = solution.y.T[:, 0]

sim_delay_exp = finderb(exp_delay, sim_delay)
sim_te_exp = sim_te[sim_delay_exp]

# normalize to zero and scale with last fit parameter:
sim_te_exp -= sim_te[0]
sim_te_exp /= np.amax(sim_te_exp)
sim_te_exp *= te_scaling_fit * 1e-3

tt_label = np.round(therm_time_fit, 1)
gep_label = np.round(gep_fit*1e2, 2)
te_scale_label = np.round(te_scaling_fit*1e-3, 4)
t0_shift_label = np.round(t0_shift_fit, 2)

plt.scatter(exp_delay, exp_te, color='blue', label=r'experimental data')
plt.plot(exp_delay, sim_te_exp, color='red', label=f'tt = {tt_label} fs \ngep={gep_label} [PW/m^3K]\nscale={te_scale_label}\nt0_shift={t0_shift_label}')
plt.xlabel(r'delay [ps]', fontsize=16)
plt.ylabel(r'Differential reflectivity [a.u.]', fontsize=16)
plt.title(f'{int(temp_0)} K', fontsize=18)
plt.legend(fontsize=14)
plt.show()

# file = 'input_data/MST/fit_figures/global_try/gloabl_fit_values.txt'
#
# param_fstring = f'{temp_0}\t{therm_time_fit}\t{gep_fit}\t{te_scaling_fit}\n'
#
# with open(file, 'a') as fit_params_file:
#     fit_params_file.write(param_fstring)
