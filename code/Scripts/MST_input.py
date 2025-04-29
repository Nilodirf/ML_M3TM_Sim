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

####### Define temperature and pulse for simulations:
temp_0 = 65.
fluence = 9.8e-3

####### Load exp data:
exp_data = np.loadtxt(f'input_data/MST/exp_data/new_experi_{int(temp_0)}k.dat')
exp_delay = exp_data[:, 0]
exp_te = exp_data[:, 1]


# Define the double exponential, give the fit parameters of the doub. exp.
# dbex_delay= ..
# dbex_te=...

###### set initial fit values:
lower_bounds = [100, 0.1, 8.5]
upper_bounds = [1300, 0.35, 10]

bounds = (lower_bounds, upper_bounds)

therm_time_initial = (upper_bounds[0]+lower_bounds[0])/2
gep_initial = (upper_bounds[1]+lower_bounds[1])/2
te_scaling_initial = (upper_bounds[2]+lower_bounds[2])/2

p0 = [therm_time_initial, gep_initial, te_scaling_initial]

####### Simulation to fit:
def fit_te_to_exp(exp_delay, therm_time_test, gep_test, te_scaling_test):

    MST = SimMaterials(name='Mn3Si2Te6', cp_max=1.28e6, cp_method='Debye', tdeb=159.,  kappap=0.,
                       ce_gamma='input_data/MST/MST_cmag.txt', gep=gep_test*1e17)

    # MST.add_phonon_subsystem(gpp=2.5e17, cp2_max=0.2e6, cp2_method='Debye')

    # Create a sample, then add desired layers of the materials you want to simulate.
    # The first material to be added will be closest to the laser pulse and so on.
    sample = SimSample()
    sample.add_layers(material=MST, layers=1,  dz=1e-9, pen_dep=1e-9)

    # Create a laser pulse with the desired parameters. (Fluence in mJ/cm^2)
    pulse = SimPulse(sample=sample, method='LB', pulse_width=25.5e-15, fluence=fluence, delay=1e-12, therm_time=therm_time_test*1e-15)

    # Initialize the simulation with starting temperature and final time, the solver to be used and the maximum timestep:
    sim = SimDynamics(sample=sample, pulse=pulse, end_time=13e-12, ini_temp=temp_0, solver='Radau', max_step=1e-12)

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
    sim_te_exp *= te_scaling_test*1e-3

    print(f"THERM_TIME: {therm_time_test}\n"
          f"GEP: {gep_test*1e2}\n"
          f"TE_SCALING: {te_scaling_test}\n")

    return sim_te_exp

####### Fit:
p_opt, p_cov = curve_fit(fit_te_to_exp, exp_delay, exp_te, p0=p0)
# popt, pcov = curve_fit(fit_te_to_exp, dbex_delay, dbex_te, p0, bounds, method)
print(p_opt, p_cov)


######## Reproduce fit:
therm_time_fit = p_opt[0]
gep_fit = p_opt[1]
te_scaling_fit = p_opt[2]

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
sim = SimDynamics(sample=sample, pulse=pulse, end_time=13e-12, ini_temp=temp_0, solver='Radau', max_step=1e-14)

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
sim_te_exp *= te_scaling_fit*1e-3

tt_label = np.round(therm_time_fit, 1)
gep_label = np.round(gep_fit*1e2, 2)
te_scale_label = np.round(te_scaling_fit*1e-3, 4)

plt.scatter(exp_delay, exp_te, color='blue', label=r'experimental data')
# plt.plot(dbex_delay, dbex_te, color='green', label=r'double exponential')
plt.plot(exp_delay, sim_te_exp, color='red', label=f'tt = {tt_label} fs \ngep={gep_label} [PW/m^3K]\nscale={te_scale_label}')
plt.xlabel(r'delay [ps]', fontsize=16)
plt.ylabel(r'Differential reflectivity [a.u.]', fontsize=16)
plt.title(f'{int(temp_0)} K', fontsize=18)
plt.legend(fontsize=14)
plt.show()




