import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.rate_functions import calculate_coulomb_logarithm

plt.rcParams.update({'font.size': 16})
plt.close('all')

num_samples = 100

n0_list = [1e18, 1e22]  # m^-3
linestyle_list = ['-', '--']
# n0_list = [1e18, 1e20, 1e22]  # m^-3
# linestyle_list = ['-', '--', ':']

for i, n0 in enumerate(n0_list):
    linestyle = linestyle_list[i]

    n = n0 * np.ones(num_samples)
    ne = n
    ni = n
    T = np.linspace(0.2, 1e4, num_samples)
    Te = T
    Ti = T

    A = 1
    Z = 2

    coulomb_log = calculate_coulomb_logarithm(ne, Te, ni, Ti, Z=Z, A=A)

    plt.figure(1)
    for key in coulomb_log:
        plt.plot(T, coulomb_log[key], linestyle=linestyle, label='n=' + str(n0) + ', ' + key)
    plt.xlabel('T [eV]')
    plt.ylabel('Coulomb Logarithm')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
