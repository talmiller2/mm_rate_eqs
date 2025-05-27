import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 12
plt.close('all')

from mm_rate_eqs.fusion_functions import load_sigma_v_fusion_files, get_E_reaction, get_E_charged, \
    get_fusion_power_multiple_ions, set_ion_densities_quasi_neutral, get_fuel_label
from mm_rate_eqs.plasma_functions import get_brem_radiation_loss_relativistic, get_cyclotron_radiation_loss, \
    define_electron_charge, get_brem_radiation_loss, define_boltzmann_constant
from mm_rate_eqs.constants_functions import define_vacuum_permeability, define_factor_eV_to_K

e = define_electron_charge()

# Ti_keV = np.linspace(1, 1000, 1000)
Ti_keV = np.logspace(0, 3, 3000)

sigma_v_dict = load_sigma_v_fusion_files(Ti_keV)

process_list = []
process_list += ['D-T']
process_list += ['D-He3']
process_list += ['p-B11']
process_list += ['pure D-D']
# process_list += ['He3-catalyzed D-D']
# process_list += ['T-catalyzed D-D']
process_list += ['fully-catalyzed D-D']

color_list = ['b', 'g', 'r', 'grey', 'k', 'k', 'k']

plt.figure(1, figsize=(8, 6))
plt.figure(2, figsize=(8, 6))
plt.figure(3, figsize=(8, 6))

for ip, process in enumerate(process_list):
    color = color_list[ip]

    print('*****', process, '*****')

    # normalize ion densities to satisfy quasi-neutrality with electrons
    # ne = 1e20  # [m^-3]
    ne = 1e21  # [m^-3]
    ions_list, Zi_list, ni_array, ni = set_ion_densities_quasi_neutral(process, ne, Ti_keV, sigma_v_dict)

    ## calculate fusion power
    P_fus_tot, P_fus_charged_tot = get_fusion_power_multiple_ions(ni_array, Ti_keV, ions_list, sigma_v_dict)

    # Te_over_Ti_list = [1, 0.1]
    # Te_over_Ti_list = [1, 0.01]
    # Te_over_Ti_list = [1, 0.1, 0.01]
    # Te_over_Ti_list = [1, 0.3, 0.1]
    # Te_over_Ti_list = [1, 0.5, 0.3]
    Te_over_Ti_list = [1, 0.4, 0.1]
    linestyle_list = ['-', '--', ':']

    for ind_Te, Te_over_Ti in enumerate(Te_over_Ti_list):
        print('   @@@ Te_over_Ti =', Te_over_Ti)
        # label = process + ' $T_e/T_i=$' + str(Te_over_Ti)
        if ind_Te == 0:
            label = get_fuel_label(process)
        else:
            label = None

        ### calculate and plot Q_fuel
        plt.figure(1)

        # B = 0.001  # [T]
        # B = 1  # [T]
        # B = 5  # [T]
        # B = 10  # [T]
        B = 20  # [T]

        Te_keV = Te_over_Ti * Ti_keV
        P_brem = get_brem_radiation_loss_relativistic(ni_array, Zi_list, Te_keV, use_relativistic_correction=False)
        P_cyc = get_cyclotron_radiation_loss(ne, Te_keV, B)

        # include_P_cyc = False
        include_P_cyc = True

        if include_P_cyc:
            P_rad = P_brem + P_cyc
            title_suffix = ' (including $P_{cyc}$)'
            # title_suffix = ''
            file_suffix = '_wPcyc'
        else:
            P_rad = P_brem  # literature version without P_cyc
            title_suffix = ' (neglecting $P_{cyc}$)'
            file_suffix = '_woPcyc'

        # ideal gas
        Ti_J = 1e3 * e * Ti_keV
        Te_J = 1e3 * e * Te_keV
        p = ne * Ti_J + ni * Te_J  # [Pa]=[J/m^3]
        E0 = 3 / 2 * p  # volumetric energy [J/m^3]

        tau = 0.1  # [s]
        # Q_fuel_old = P_fus_tot / (P_rad + E0 / tau)
        Q_fuel = P_fus_tot / (P_rad - P_fus_charged_tot + E0 / tau)
        Q_fuel[Q_fuel < 0] = np.nan

        plt.plot(Ti_keV, Q_fuel,
                 color=color,
                 linestyle=linestyle_list[ind_Te],
                 label=label)

        ### calculate and plot p*tau
        const_Q_fuel = np.inf
        # const_Q_fuel = 100
        # const_Q_fuel = 1
        tau_for_const_Q_fuel = E0 / (P_fus_tot / const_Q_fuel + P_fus_charged_tot - P_rad)
        p_tau = p * tau_for_const_Q_fuel
        p_tau[p_tau < 0] = np.nan
        p_tau_keV = p_tau / (1e3 * e)

        ### calculate and plot plasma beta
        mu0 = define_vacuum_permeability()
        p_magnetic = B ** 2 / (2 * mu0)  # [Pa]=[J/m^3]
        plasma_beta = p / p_magnetic

        if np.all(np.isnan(p_tau_keV)):
            # label = None
            pass
        else:
            ind_min_p_tau = np.nanargmin(p_tau_keV)
            T_min_p_tau = Ti_keV[ind_min_p_tau]
            print('      T_min_p_tau=', T_min_p_tau, '[keV]')
            print('      p_tau_keV=', p_tau_keV[ind_min_p_tau], '[m^-3 keV s]')
            print('      plasma_beta @opt=', plasma_beta[ind_min_p_tau])
            print('      P_fus_tot @opt=', P_fus_tot[ind_min_p_tau] / 1e6, '[MW/m^3]')
            print('      tau @opt=', tau_for_const_Q_fuel[ind_min_p_tau], '[s]')

            plt.figure(2)
            plt.scatter(Ti_keV[ind_min_p_tau], p_tau_keV[ind_min_p_tau], color=color)

        plt.figure(2)
        plt.plot(Ti_keV, p_tau_keV,
                 color=color,
                 linestyle=linestyle_list[ind_Te],
                 label=label)

        plt.figure(3)
        plt.plot(Ti_keV, plasma_beta,
                 color=color,
                 linestyle=linestyle_list[ind_Te],
                 label=label)

plt.figure(1)
title = '$n_e=10^{' + str(int(np.log10(ne))) + '}[m^{-3}], \\tau=10^{' + str(int(np.log10(tau))) + '}[s]$'
if include_P_cyc:
    title += ', B=' + str(B) + '[T]'
title += title_suffix
plt.title(title)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$Q_{fuel}$')
plt.xlim([min(Ti_keV), max(Ti_keV)])
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(2)
title = '$n_e=10^{' + str(int(np.log10(ne))) + '}[m^{-3}], Q_{fuel}=$' + str(const_Q_fuel)
if include_P_cyc:
    title += ', B=' + str(B) + '[T]'
title += title_suffix
plt.title(title)
plt.xscale('log')
plt.yscale('log')
plt.xlim([min(Ti_keV), max(Ti_keV)])
plt.ylim([1e21, 1e26])
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$p \cdot \\tau$ [m$^{-3}$keV s]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(3)
plt.title('$n_e=10^{' + str(int(np.log10(ne))) + '}[m^{-3}], B=' + str(B) + '[T]$' + title_suffix)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$\\beta = p / p_{mag}$')
plt.xlim([min(Ti_keV), max(Ti_keV)])
plt.legend()
plt.grid(True)
plt.tight_layout()

# ## save figs at higher res
# figs_folder = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/'
# plt.figure(1)
# plt.savefig(figs_folder + 'Q_fuel_at_const_ne' + file_suffix + '.png', format='png', dpi=600)
# plt.figure(2)
# plt.savefig(figs_folder + 'lawson_p_tau_at_const_ne' + file_suffix + '.png', format='png', dpi=600)
# plt.figure(3)
# plt.savefig(figs_folder + 'beta_at_const_ne' + file_suffix + '.png', format='png', dpi=600)
