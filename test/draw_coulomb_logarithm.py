import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.rate_functions import calculate_coulomb_logarithm

plt.rcParams.update({'font.size': 16})
plt.close('all')

num_samples = 100

n0_list = [1e15, 1e18, 1e22]  # m^-3
linestyle_list = ['-', '--', ':']
color_list = ['b', 'r', 'g']
# n0_list = [1e18, 1e20, 1e22]  # m^-3
# linestyle_list = ['-', '--', ':']

for i, n0 in enumerate(n0_list):
    # linestyle = linestyle_list[i]
    color = color_list[i]

    n = n0 * np.ones(num_samples)
    ne = n
    ni = n
    # T = np.linspace(0.2, 1e4, num_samples)
    T = np.linspace(0.01, 1e4, num_samples)
    Te = T
    Ti = T

    A = 1
    Z = 2

    coulomb_log = calculate_coulomb_logarithm(ne, Te, ni, Ti, Z=Z, A=A)

    plt.figure(1)
    # scattering_type_list = coulomb_log.keys()
    # scattering_type_list = ['ii', 'ee']
    scattering_type_list = ['ie_overheated_ions', 'ie_cold_electrons', 'ie_hot_electrons']
    for j, key in enumerate(scattering_type_list):
        linestyle = linestyle_list[j]
        plt.plot(T, coulomb_log[key], linestyle=linestyle, color=color,
                 label='n=' + '{:.0e}'.format(n0) + ', ' + key)
    plt.xlabel('T [eV]')
    plt.ylabel('Coulomb Logarithm')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
