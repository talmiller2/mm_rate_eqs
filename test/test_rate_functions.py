import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.rate_functions import calculate_transition_density, \
    get_isentrope_temperature, get_thermal_velocity, \
    get_coulomb_scattering_rate, get_mirror_cell_sizes, \
    get_transmission_rate, calculate_mean_free_path

### Plot different parameters
settings = define_default_settings()
n0 = settings['n0']
Ti_0 = settings['Ti_0']
Te_0 = settings['Te_0']

mfp_i = calculate_mean_free_path(n0, Ti_0, Te_0, settings, species='ions')
mfp_e = calculate_mean_free_path(n0, Ti_0, Te_0, settings, species='electrons')
print('mfp_i=', '{:.3e}'.format(mfp_i), 'm')
print('mfp_e=', '{:.3e}'.format(mfp_e), 'm')
n_trans = calculate_transition_density(n0, Ti_0, Te_0, settings)
print('n_trans=', '{:.3e}'.format(n_trans), 'm^-3')

# plot isentropes
plt.close('all')
n_array = np.linspace(n0 * 1e-3, n0, 1000)
Ti_isentrope_array = get_isentrope_temperature(n_array, settings, species='ions')
Te_isentrope_array = get_isentrope_temperature(n_array, settings, species='electrons')
plt.figure(100)
plt.plot(n_array, Ti_isentrope_array / settings['keV'], label='i', color='r')
plt.plot(n_array, Te_isentrope_array / settings['keV'], label='e', color='b')
plt.legend()
plt.xlabel('n [g/cc]')
plt.ylabel('T [keV]')
plt.title('Isentropes')
plt.tight_layout()
plt.grid()

# plot rates
v_th_i = get_thermal_velocity(Ti_isentrope_array, settings, species='ions')
v_th_e = get_thermal_velocity(Te_isentrope_array, settings, species='electrons')
mirror_cell_sizes = get_mirror_cell_sizes(n_array, Ti_isentrope_array, Te_isentrope_array, settings)
nu_i_trans = get_transmission_rate(v_th_i, mirror_cell_sizes)
nu_e_trans = get_transmission_rate(v_th_e, mirror_cell_sizes)
nu_i_scat = get_coulomb_scattering_rate(n_array, Ti_isentrope_array, Te_isentrope_array, settings, species='ions')
nu_e_scat = get_coulomb_scattering_rate(n_array, Ti_isentrope_array, Te_isentrope_array, settings, species='electrons')
plt.figure(101)
plt.plot(n_array, nu_i_scat, '-', label='i scat', color='r')
plt.plot(n_array, nu_i_trans, '--', label='i trans', color='r')
plt.plot(n_array, nu_e_scat, '-', label='e scat', color='b')
plt.plot(n_array, nu_e_trans, '--', label='e trans', color='b')
plt.legend()
plt.xlabel('n [g/cc]')
plt.ylabel('rate [$s^{-1}$]')
plt.title('Rates')
plt.tight_layout()
plt.yscale('log')
plt.grid()

# plot mfps
mfp_i_array = calculate_mean_free_path(n_array, Ti_isentrope_array, Te_isentrope_array, settings, species='ions')
mfp_e_array = calculate_mean_free_path(n_array, Ti_isentrope_array, Te_isentrope_array, settings, species='electrons')
plt.figure(102)
plt.plot(n_array, mfp_i_array, '-', label='i', color='r')
plt.plot(n_array, mfp_e_array, '-', label='e', color='b')
plt.legend()
plt.xlabel('n [g/cc]')
plt.ylabel('mfp [m]')
plt.title('Mean free path')
plt.tight_layout()
plt.grid()

# test mfp calc for different gasses
for gas_name in ['hydrogen', 'helium', 'lithium', 'sodium', 'potassium']:
    settings = define_default_settings({'gas_name': gas_name})
    n = 1e17
    # n = 3e16
    T = 2400 / settings['eV_to_K']
    # T = 0.3
    # T = 1100 / settings['eV_to_K']
    # n = 1.3e17
    # T = 780 / settings['eV_to_K']
    mfp = calculate_mean_free_path(n, T, T, settings)
    print(gas_name + ' mfp = ' + str(mfp) + ' m')
