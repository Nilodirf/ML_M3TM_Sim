import numpy as np
from matplotlib import pyplot as plt

from code.Source.finderb import finderb


def c_e(t):
    gamma = 619

    return gamma*t


def c_m(t):
    cm_dat = np.loadtxt('input_data/MST/MST_cmag.txt')

    temps = cm_dat[:, 0]
    cm = cm_dat[:, 1]

    temps_t = finderb(t, temps)
    cm_t = cm[temps_t]

    return cm_t


def c_p(t):

    tdeb = 59.
    cp_max = 1.36e6

    t_red = tdeb / t

    heat_capacities = np.zeros_like(t)

    for i, yi in enumerate(t_red):
        D_yi = 3 * debye_integral(yi) / yi ** 3 if yi != 0 else 0
        heat_capacities[i] = cp_max * D_yi

    return heat_capacities


def debye_integral(t_red, num_points=1000):

    if t_red == 0:
        return 0

    x = np.linspace(1e-5, t_red, num_points)
    integrand = x ** 3 / (np.exp(x) - 1)
    integral_approx = np.trapz(integrand, x)

    return integral_approx

temp_grid = np.arange(0, 300, 1e-2)

ce = c_e(temp_grid)*1e-6
cm = c_m(temp_grid)*1e-6
cp = c_p(temp_grid)*1e-6

plt.figure(figsize=(8, 6))

plt.plot(temp_grid, ce + cm + cp, ls='dashed', lw=2.0, label=r'total', color='black')
plt.plot(temp_grid, cp, lw=2.0, label=r'lattice', color='blue')
plt.plot(temp_grid, ce + cm, lw=2.0, color='green', label=r'electrons and spins')

plt.xlim(0, 300)
plt.ylim(0, 1.48)

plt.xlabel(r'T [K]', fontsize=16)
plt.ylabel(r'C [MJ/m$^3$K]', fontsize=16)

plt.legend(fontsize=14)

plt.savefig('MST_heat_capac.pdf')
plt.show()
