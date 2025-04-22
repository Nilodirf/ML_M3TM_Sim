# All changes to redefine parameters of the simulation can be done in this file.
# For documentation of the simulation methods or class parameters, see the respective files.
# Unless explicitly stated otherwise, all parameters are to be put in SI units.
# Short documentation of the simulation setup is given before each block here.

from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt

# Import classes from other files to set up materials, sample, pulse and the dynamical functions:
from code.Source.mats import SimMaterials
from code.Source.sample import SimSample
from code.Source.pulse import SimPulse
from code.Source.mainsim import SimDynamics
from code.Source.finderb import finderb

####### Load exp data:
exp_data = np.loadtxt('input_data/MST/exp_data/new_experi_75k.dat')
exp_delay = exp_data[:, 0]
exp_te = exp_data[:, 1]

###### set initial fit values:
therm_time_initial = 0.5e-12
gep_initial = 2.5e17
te_scaling_initial = 0.009

p0 = [therm_time_initial, gep_initial, te_scaling_initial]

lower_bounds = [100e-15, 2e17, 0.006]
upper_bounds = [2e-12, 4e17, 0.01]
bounds = (lower_bounds, upper_bounds)

####### Simulation to fit:
def fit_te_to_exp(exp_delay, therm_time_test, gep_test, te_scaling_test):

    MST = SimMaterials(name='Mn3Si2Te6', cp_max=1.28e6, cp_method='Debye', tdeb=159.,  kappap=0.,
                       ce_gamma='input_data/MST/MST_cmag.txt', gep=gep_test)

    # MST.add_phonon_subsystem(gpp=2.5e17, cp2_max=0.2e6, cp2_method='Debye')

    # Create a sample, then add desired layers of the materials you want to simulate.
    # The first material to be added will be closest to the laser pulse and so on.
    sample = SimSample()
    sample.add_layers(material=MST, layers=1,  dz=1e-9, pen_dep=1e-9)

    # Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
    pulse = SimPulse(sample=sample, method='LB', pulse_width=25.5e-15, fluence=9.8e-3, delay=1e-12, therm_time=therm_time_test)

    # Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
    sim = SimDynamics(sample=sample, pulse=pulse, end_time=13e-12, ini_temp=75., solver='Radau', max_step=1e-14)

    # Run the simulation by calling the function that creates the map of all three baths
    solution = sim.get_t_m_maps()

    # extract te and use binary search to fine te at the experimental delays
    sim_delay = solution.t*1e12-1
    sim_te = solution.y.T[:, 0]

    sim_delay_exp = finderb(exp_delay, sim_delay)
    sim_te_exp = sim_te[sim_delay_exp]

    # normalize to zero and scale with last fit parameter:
    sim_te_exp -= sim_te[0]
    sim_te_exp /= np.amax(sim_te_exp)
    sim_te_exp *= te_scaling_test

    print(f"THERM_TIME: {therm_time_test}\n"
          f"GEP: {gep_test}\n"
          f"TE_SCALING: {te_scaling_test}\n")

    return sim_te_exp

####### Fit:
p_opt, p_cov = curve_fit(fit_te_to_exp, exp_delay, exp_te, p0=p0, bounds=bounds)
print(p_opt, p_cov)



######## Reproduce fit:
therm_time_fit = p_opt[0]
gep_fit = p_opt[1]
te_scaling_fit = p_opt[2]

MST = SimMaterials(name='Mn3Si2Te6', cp_max=1.28e6, cp_method='Debye', tdeb=159.,  kappap=0.,
                       ce_gamma='input_data/MST/MST_cmag.txt', gep=gep_fit)

# MST.add_phonon_subsystem(gpp=2.5e17, cp2_max=0.2e6, cp2_method='Debye')

# Create a sample, then add desired layers of the materials you want to simulate.
# The first material to be added will be closest to the laser pulse and so on.
sample = SimSample()
sample.add_layers(material=MST, layers=1,  dz=1e-9, pen_dep=1e-9)

# Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
pulse = SimPulse(sample=sample, method='LB', pulse_width=25.5e-15, fluence=9.8e-3, delay=1e-12, therm_time=therm_time_fit)

# Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
sim = SimDynamics(sample=sample, pulse=pulse, end_time=13e-12, ini_temp=25., solver='Radau', max_step=1e-14)

# Run the simulation by calling the function that creates the map of all three baths
solution = sim.get_t_m_maps()

# extract te and use binary search to fine te at the experimental delays
sim_delay = solution.t*1e12-1
sim_te = solution.y.T[:, 0]

sim_delay_exp = finderb(exp_delay, sim_delay)
sim_te_exp = sim_te[sim_delay_exp]

# normalize to zero and scale with last fit parameter:
sim_te_exp -= sim_te[0]
sim_te_exp /= np.amax(sim_te_exp)
sim_te_exp *= te_scaling_fit

plt.plot(exp_delay, sim_te_exp)
plt.scatter(exp_delay, exp_te)
plt.show()




