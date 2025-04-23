import numpy as np
from matplotlib import pyplot as plt

file = 'second_try/T_tt_gep_scale.txt'

content = np.loadtxt(file)

temps = content[:, 0]
tt = content[:, 1]
gep = content[:, 2]
scale = content[:, 3]

plt.scatter(temps, tt, s=50, color='blue')
plt.plot(temps, tt, color='blue')
plt.xlabel(r'T [K]', fontsize=16)
plt.ylabel(r'thermalization time [fs]', fontsize=16)
plt.show()

plt.scatter(temps, gep, s=50, color='red')
plt.plot(temps, gep, color='red')
plt.xlabel(r'T [K]', fontsize=16)
plt.ylabel(r'g$_{ep}$ [PW/m^3/K]', fontsize=16)
plt.show()

plt.scatter(temps, scale, s=50, color='green')
plt.plot(temps, scale, color='green')
plt.xlabel(r'T [K]', fontsize=16)
plt.ylabel(r'scale [a.u.]', fontsize=16)
plt.show()
