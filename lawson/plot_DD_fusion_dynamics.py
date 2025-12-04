import numpy as np
from matplotlib import pyplot as plt

# plt.rcParams['font.size'] = 12
plt.rcParams['font.size'] = 14
plt.close('all')

from mm_rate_eqs.fusion_functions import load_sigma_v_fusion_files, get_E_reaction, get_E_charged, \
    get_fusion_power_multiple_ions, set_ion_densities_quasi_neutral, update_ion_latex_name
from mm_rate_eqs.plasma_functions import get_brem_radiation_loss_relativistic, get_cyclotron_radiation_loss, \
    define_electron_charge, get_brem_radiation_loss, define_boltzmann_constant
from mm_rate_eqs.constants_functions import define_vacuum_permeability, define_factor_eV_to_K

e = define_electron_charge()
MeV_to_J = e * 1e6

Ti_keV = 50
# Ti_keV = 15
ni_0 = 1e21  # [m^-3]

sigma_v_dict = load_sigma_v_fusion_files(Ti_keV)

### ideal steady state solution for reference
ions_list, Zi_list, ni_array_ideal, ni = set_ion_densities_quasi_neutral('fully-catalyzed D-D', np.array([ni_0]),
                                                                         np.array([Ti_keV]))
P_fus_ideal, P_fus_ch_ideal, _ = get_fusion_power_multiple_ions(ni_array_ideal, Ti_keV, ions_list, sigma_v_dict)
###

ions = ['D', 'T', 'He3']
n = {'D': [ni_0], 'T': [0], 'He3': [0]}
# n = {'D': [ni_array_ideal[0]], 'T': [0], 'He3': [0]}
# n = {'D': [0], 'T': [0], 'He3': [ni_0]}
# n = {'D': [0], 'T': [ni_0], 'He3': [0]}
# n = {'D': [ni_0 / 2], 'T': [ni_0 / 2], 'He3': [0]}
# n = {'D': [ni_array_ideal[0]], 'T': [ni_array_ideal[1]], 'He3': [ni_array_ideal[2]]}

P_fus = [np.nan]
P_fus_ch = [np.nan]
E_fus = [0]
E_fus_ch = [0]

n['D_pure'] = [n['D'][-1]]
P_fus_pure = [np.nan]
P_fus_ch_pure = [np.nan]
E_fus_pure = [0]
E_fus_ch_pure = [0]

# dt = 1e-3
# tmax = 1
dt = 0.01
tmax = 10
# tmax = 100
t = np.arange(0, tmax, dt)  # [s]

for ind_t in range(1, len(t)):
    rate_DD_pT = 0.5 * (n['D'][-1]) ** 2 * sigma_v_dict['D_D_to_p_T']
    rate_DD_nHe = 0.5 * (n['D'][-1]) ** 2 * sigma_v_dict['D_D_to_n_He3']
    rate_DD = rate_DD_pT + rate_DD_nHe
    power_DD = (rate_DD_pT * get_E_reaction(reaction='D_D_to_p_T') * MeV_to_J
                + rate_DD_nHe * get_E_reaction(reaction='D_D_to_n_He3') * MeV_to_J)
    power_ch_DD = (rate_DD_pT * get_E_charged(reaction='D_D_to_p_T') * MeV_to_J
                   + rate_DD_nHe * get_E_charged(reaction='D_D_to_n_He3') * MeV_to_J)

    rate_DT = n['D'][-1] * n['T'][-1] * sigma_v_dict['D_T_to_n_alpha']
    power_DT = rate_DT * get_E_reaction(reaction='D_T_to_n_alpha') * MeV_to_J
    power_ch_DT = rate_DT * get_E_charged(reaction='D_T_to_n_alpha') * MeV_to_J

    rate_DHe3 = n['D'][-1] * n['He3'][-1] * sigma_v_dict['D_He3_to_p_alpha']
    power_DHe3 = rate_DHe3 * get_E_reaction(reaction='D_He3_to_p_alpha') * MeV_to_J
    power_ch_DHe3 = rate_DHe3 * get_E_charged(reaction='D_He3_to_p_alpha') * MeV_to_J

    n['D'] += [n['D'][-1] + (- rate_DD_pT - rate_DD_nHe - rate_DT) * dt]
    # n['D'] += [ni_0]
    n['T'] += [n['T'][-1] + (rate_DD_pT - rate_DT) * dt]
    # n['T'] += [0]
    n['He3'] += [n['He3'][-1] + (rate_DD_nHe - rate_DHe3) * dt]
    # n['He3'] += [0]

    P_fus += [power_DD + power_DT + power_DHe3]
    E_fus += [E_fus_ch[-1] + dt * P_fus[-1]]
    P_fus_ch += [power_ch_DD + power_ch_DT + power_ch_DHe3]
    E_fus_ch += [E_fus_ch[-1] + dt * P_fus_ch[-1]]

    ## pure D-D
    rate_pure_DD_pT = 0.5 * (n['D_pure'][-1]) ** 2 * sigma_v_dict['D_D_to_p_T']
    rate_pure_DD_nHe = 0.5 * (n['D_pure'][-1]) ** 2 * sigma_v_dict['D_D_to_n_He3']
    power_pure_DD = (rate_pure_DD_pT * get_E_reaction(reaction='D_D_to_p_T') * MeV_to_J
                     + rate_pure_DD_nHe * get_E_reaction(reaction='D_D_to_n_He3') * MeV_to_J)
    power_ch_pure_DD = (rate_pure_DD_pT * get_E_charged(reaction='D_D_to_p_T') * MeV_to_J
                        + rate_pure_DD_nHe * get_E_charged(reaction='D_D_to_n_He3') * MeV_to_J)

    n['D_pure'] += [n['D_pure'][-1] + (- rate_pure_DD_pT - rate_pure_DD_nHe) * dt]

    P_fus_pure += [power_pure_DD]
    E_fus_pure += [E_fus_pure[-1] + dt * P_fus_pure[-1]]
    P_fus_ch_pure += [power_ch_pure_DD]
    E_fus_ch_pure += [E_fus_ch_pure[-1] + dt * P_fus_ch_pure[-1]]

for ion in ions:
    n[ion] = np.array(n[ion])
    n[ion] /= ni_0
# print('E_fus_tot =', E_fus_tot)
# print('E_fus_charged_tot =', E_fus_charged_tot)
ni_array_ideal /= ni_0

# plt.figure(figsize=(8, 6))
plt.figure(1, figsize=(12, 6))
plt.subplot(1, 2, 1)
# plt.plot(t, n['D_pure'], label='D pure', color='b', linestyle='--')
plt.plot(t, n['D'], label='D', color='b')
plt.plot(t, n['T'], label='T', color='r')
plt.plot(t, n['He3'], label='He3', color='g')

plt.plot(t, ni_array_ideal[0] + 0 * t, label='D (ideal)', color='b', linestyle='--')
plt.plot(t, ni_array_ideal[1] + 0 * t, label='T (ideal)', color='r', linestyle='--')
plt.plot(t, ni_array_ideal[2] + 0 * t, label='He3 (ideal)', color='g', linestyle='--')

plt.xlabel('t [s]')
# plt.ylabel('n [$m^{-3}$]')
plt.ylabel('$n / n_{i,0}$')
plt.xlim([0, tmax])
# plt.ylim([ni_0 / 1e3, ni_0])
plt.ylim([1e-3, 1])
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()

# plt.figure(figsize=(8, 6))
# plt.plot(t, n['D'] / n['D'], label='D', color='b')
# plt.plot(t, n['T'] / n['D'], label='T', color='r')
# plt.plot(t, n['He3']/ n['D'], label='He3', color='g')
# plt.xlabel('t [s]')
# plt.ylabel('n / n(D)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 2)
plt.plot(t, P_fus_pure, label='pure-DD', color='b', linestyle='--')
plt.plot(t, P_fus, label='cat-DD', color='b')
# plt.plot(t, 0 * t + P_fus_ideal[0], label='ideal', color='b', linestyle=':')
plt.axhline(P_fus_ideal[0], label='cat-DD (ideal)', color='k')
# plt.plot(t, P_fus_ch, color='r')
# plt.plot(t, P_fus_ch_pure, color='r', linestyle='--')
# plt.plot(t, 0 * t + P_fus_ch_ideal[0], color='r', linestyle=':')
plt.xlabel('t [s]')
plt.ylabel('fusion power [$W/m^3$]')
# plt.suptitle('D-D pulse dynamics @ $T_i=$' + str(Ti_keV) + 'keV' + ', $n_{i,0}=$' + str(ni_0) + '$m^{-3}$')
plt.suptitle('D-D pulse dynamics @ $T_i=$' + str(Ti_keV) + 'keV' + ', $n_{i,0}=10^{21}m^{-3}$')
# plt.suptitle('D-T pulse dynamics @ $T_i=$' + str(Ti_keV) + 'keV')
plt.xlim([0, tmax])
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()

# plt.figure(2, figsize=(8, 6))
# plt.plot(t, E_fus, label='cat-DD', color='b')
# plt.plot(t, E_fus_pure, label='pure-DD', color='b', linestyle='--')
# plt.xlabel('t [s]')
# plt.ylabel('E [$J/m^3$]')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# ## save figs at higher res
figs_folder = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/'
plt.figure(1)
# plt.savefig(figs_folder + '/DD_pulse_dynamics.pdf', format='pdf')
