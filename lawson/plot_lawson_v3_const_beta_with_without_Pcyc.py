import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 12
plt.close('all')

from mm_rate_eqs.fusion_functions import load_sigma_v_fusion_files, get_E_reaction, get_E_charged, \
    get_fusion_power_multiple_ions, set_ion_densities_quasi_neutral, update_ion_latex_name
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

# color_list = ['b', 'g', 'r', 'grey', 'k', 'k', 'k']
color_list = ['b', 'g', 'r', 'orange', 'k', 'k', 'k']

# split_to_subplots = False
split_to_subplots = True
# inds_process_to_subplot = [1, 2, 3, 4, 4]
inds_process_to_subplot = [1, 2, 3, 3, 4]

if split_to_subplots:
    # figsize = (12, 7)
    figsize = (14, 9)
    plt.figure(1, figsize=figsize)
    plt.figure(2, figsize=figsize)
    plt.figure(3, figsize=figsize)
    plt.figure(4, figsize=figsize)
    inds_subplots = [1, 2, 3, 4]
else:
    # figsize = (8, 6)
    figsize = (12, 8)
    plt.figure(1, figsize=figsize)
    plt.figure(2, figsize=figsize)
    plt.figure(3, figsize=figsize)
    plt.figure(4, figsize=figsize)
    inds_subplots = [1]

for ip, process in enumerate(process_list):
    color = color_list[ip]

    print('*****', process, '*****')

    # Te_over_Ti_list = [1]
    # Te_over_Ti_list = [0.1]
    # Te_over_Ti_list = [0.01]
    # Te_over_Ti_list = [1, 0.1]
    # Te_over_Ti_list = [1, 0.01]
    # Te_over_Ti_list = [1, 0.1, 0.01]
    # Te_over_Ti_list = [1, 0.3, 0.1]
    Te_over_Ti_list = [1, 0.4, 0.1]
    linestyle_list = ['-', '--', ':']
    hatch_list = ['////', '***', '...']

    for ind_Te, Te_over_Ti in enumerate(Te_over_Ti_list):
        print('   @@@ Te_over_Ti =', Te_over_Ti)
        # label = process + ' $T_e/T_i=$' + str(Te_over_Ti)

        # if ind_Te == 0:
        #     label = update_ion_latex_name(process)
        # else:
        #     label = None
        label = update_ion_latex_name(process) + ', $T_e/T_i=' + str(Te_over_Ti) + '$'

        Te_keV = Te_over_Ti * Ti_keV

        # normalize ion densities to satisfy quasi-neutrality with electrons
        ne = 1 * Ti_keV  # dummy value, will get scales later
        ions_list, Zi_list, ni_array, ni = set_ion_densities_quasi_neutral(process, ne, Ti_keV, sigma_v_dict)

        ## rescale densities to get constant plasma_beta
        # plasma_beta_target = 0.01
        # plasma_beta_target = 0.05
        # plasma_beta_target = 0.1
        plasma_beta_target = 0.5
        # plasma_beta_target = 0.9
        # B = 0.0001  # [T]
        # B = 1  # [T]
        # B = 5  # [T]
        # B = 10  # [T]
        B = 20  # [T]

        # ideal gas
        Ti_J = 1e3 * e * Ti_keV
        Te_J = 1e3 * e * Te_keV
        p = ne * Ti_J + ni * Te_J  # [Pa]=[J/m^3]
        mu0 = define_vacuum_permeability()
        p_magnetic = B ** 2 / (2 * mu0)  # [Pa]=[J/m^3]
        plasma_beta = p / p_magnetic
        density_scale_factor = plasma_beta_target / plasma_beta
        ni *= density_scale_factor
        ni_array *= density_scale_factor
        ne *= density_scale_factor

        ## plot the density constrained by plasma beta
        plt.figure(4)
        if split_to_subplots:
            plt.subplot(2, 2, inds_process_to_subplot[ip])
        plt.plot(Ti_keV, ne,
                 color=color,
                 linestyle=linestyle_list[ind_Te],
                 label=label)

        ## calculate fusion power
        P_fus_tot, P_fus_charged_tot, reaction_rate_tot = get_fusion_power_multiple_ions(ni_array, Ti_keV, ions_list,
                                                                                         sigma_v_dict)

        ## calculate radiation losses
        P_brem = get_brem_radiation_loss_relativistic(ni_array, Zi_list, Te_keV, use_relativistic_correction=True)
        # P_cyc = get_cyclotron_radiation_loss(ne, Te_keV, B, version='Stacey')
        P_cyc = get_cyclotron_radiation_loss(ne, Te_keV, B, version='Wiedemann', r=0)
        # P_cyc = get_cyclotron_radiation_loss(ne, Te_keV, B, version='Kukushkin', r=0)

        title_suffix = ''

        P_rad = P_brem  # literature version without P_cyc
        P_rad_wcyc = P_brem + P_cyc

        ### calculate and plot Q_fuel

        # ideal gas
        Ti_J = 1e3 * e * Ti_keV
        Te_J = 1e3 * e * Te_keV
        p = ne * Ti_J + ni * Te_J  # [Pa]=[J/m^3]
        E0 = 3 / 2 * p  # volumetric energy [J/m^3]

        tau = 0.1  # [s]
        # Q_fuel_old = P_fus_tot / (P_rad + E0 / tau)
        Q_fuel = P_fus_tot / (P_rad - P_fus_charged_tot + E0 / tau)
        Q_fuel[Q_fuel < 0] = np.nan

        Q_fuel_wcyc = P_fus_tot / (P_rad_wcyc - P_fus_charged_tot + E0 / tau)
        Q_fuel_wcyc[Q_fuel_wcyc < 0] = np.nan

        plt.figure(1)
        if split_to_subplots:
            plt.subplot(2, 2, inds_process_to_subplot[ip])
        if not np.all(np.isnan(Q_fuel)):
            plt.plot(Ti_keV, Q_fuel,
                     color=color,
                     linestyle=linestyle_list[ind_Te],
                     label=label)
            # plt.plot(Ti_keV, Q_fuel_wcyc,
            #          color=color,
            #          linestyle=linestyle_list[ind_Te],
            #          )
            plt.fill_between(Ti_keV, Q_fuel, Q_fuel_wcyc,
                             color=color,
                             hatch=hatch_list[ind_Te],
                             alpha=0.3)

        ### calculate and plot p*tau
        const_Q_fuel = np.inf
        # const_Q_fuel = 100
        # const_Q_fuel = 1
        tau_for_const_Q_fuel = E0 / (P_fus_tot / const_Q_fuel + P_fus_charged_tot - P_rad)
        tau_for_const_Q_fuel[tau_for_const_Q_fuel < 0] = np.nan
        p_tau = p * tau_for_const_Q_fuel
        p_tau_keV = p_tau / (1e3 * e)

        tau_for_const_Q_fuel_wcyc = E0 / (P_fus_tot / const_Q_fuel + P_fus_charged_tot - P_rad_wcyc)
        tau_for_const_Q_fuel_wcyc[tau_for_const_Q_fuel_wcyc < 0] = np.nan
        p_tau_wcyc = p * tau_for_const_Q_fuel_wcyc
        p_tau_keV_wcyc = p_tau_wcyc / (1e3 * e)

        # optimal point without cyc
        if not np.all(np.isnan(p_tau_keV)):
            ind_min_p_tau = np.nanargmin(p_tau_keV)
            T_min_p_tau = Ti_keV[ind_min_p_tau]

            print('      T_min_p_tau=', T_min_p_tau, '[keV]')
            print('      p_tau_keV=', p_tau_keV[ind_min_p_tau], '[m^-3 keV s]')
            print('      ne @opt=', ne[ind_min_p_tau], '[m$^{-3}$]')
            print('      P_fus_tot @opt=', P_fus_tot[ind_min_p_tau] / 1e6, '[MW/m^3]')
            # label += ', $P_{fus}$=' + '{:.1f}'.format(P_fus_tot[ind_min_p_tau] / 1e6) + '$[MW/m^3]$'
            label += ', $P_{fus}$=' + '{:.1f}'.format(P_fus_tot[ind_min_p_tau] / 1e6)
            print('      tau @opt=', tau_for_const_Q_fuel[ind_min_p_tau], '[s]')
            T = 0.1 * ni[ind_min_p_tau] / reaction_rate_tot[ind_min_p_tau]
            print('      T(10% fuel depletion) @opt=', T, '[s]')
            print('      T/tau=', T / tau_for_const_Q_fuel[ind_min_p_tau])

            plt.figure(2)
            if split_to_subplots:
                plt.subplot(2, 2, inds_process_to_subplot[ip])
            plt.scatter(Ti_keV[ind_min_p_tau], p_tau_keV[ind_min_p_tau], color=color)

            plt.figure(3)
            if split_to_subplots:
                plt.subplot(2, 2, inds_process_to_subplot[ip])
            plt.scatter(Ti_keV[ind_min_p_tau], tau_for_const_Q_fuel[ind_min_p_tau], color=color)

        # optimal point with cyc
        if not np.all(np.isnan(p_tau_keV_wcyc)):
            ind_min_p_tau_wcyc = np.nanargmin(p_tau_keV_wcyc)
            T_min_p_tau_wcyc = Ti_keV[ind_min_p_tau_wcyc]
            # label += '/' + '{:.0f}'.format(P_fus_tot[ind_min_p_tau_wcyc] / 1e6) # the fusion power without/with cyc in [MW/m^3]

            plt.figure(2)
            if split_to_subplots:
                plt.subplot(2, 2, inds_process_to_subplot[ip])
            plt.scatter(Ti_keV[ind_min_p_tau_wcyc], p_tau_keV_wcyc[ind_min_p_tau_wcyc], color=color,
                        facecolor='none')

            plt.figure(3)
            if split_to_subplots:
                plt.subplot(2, 2, inds_process_to_subplot[ip])
            plt.scatter(Ti_keV[ind_min_p_tau_wcyc], tau_for_const_Q_fuel_wcyc[ind_min_p_tau_wcyc], color=color,
                        facecolor='none')

        else:
            label += '/NaN'  # no valid solution with cyc so indicate no power

        plt.figure(2)
        if split_to_subplots:
            plt.subplot(2, 2, inds_process_to_subplot[ip])
        if not np.all(np.isnan(p_tau_keV)):
            plt.plot(Ti_keV, p_tau_keV,
                     color=color,
                     linestyle=linestyle_list[ind_Te],
                     label=label)
            # plt.plot(Ti_keV, p_tau_keV_wcyc,
            #          color=color,
            #          linestyle=linestyle_list[ind_Te])
            plt.fill_between(Ti_keV, p_tau_keV, p_tau_keV_wcyc,
                             color=color,
                             hatch=hatch_list[ind_Te],
                             alpha=0.3)

        plt.figure(3)
        if split_to_subplots:
            plt.subplot(2, 2, inds_process_to_subplot[ip])
        if not np.all(np.isnan(tau_for_const_Q_fuel)):
            plt.plot(Ti_keV, tau_for_const_Q_fuel,
                     color=color,
                     linestyle=linestyle_list[ind_Te],
                     label=label)
            # plt.plot(Ti_keV, tau_for_const_Q_fuel_wcyc,
            #          color=color,
            #          linestyle=linestyle_list[ind_Te])
            plt.fill_between(Ti_keV, tau_for_const_Q_fuel, tau_for_const_Q_fuel_wcyc,
                             color=color,
                             hatch=hatch_list[ind_Te],
                             alpha=0.3,
                             )

    # compare to 10% depletion time
    plt.figure(3)
    if split_to_subplots:
        plt.subplot(2, 2, inds_process_to_subplot[ip])
    T_depletion10 = 0.1 * ni / reaction_rate_tot
    plt.plot(Ti_keV, T_depletion10,
             color=color,
             # linestyle='--',
             # label=label + ' $T_{10\%}$',
             label=update_ion_latex_name(process) + ' 10%-depletion',
             linewidth=3,
             # alpha=0.5,
             )

## plot aesthetics
plt.figure(1)
title = 'Lawson Criterion:  $B=$' + str(B) + '[T], $\\beta=$' + str(plasma_beta_target)
title += ', $\\tau=10^{' + str(int(np.log10(tau))) + '}[s]$' + title_suffix
if split_to_subplots:
    plt.suptitle(title)
else:
    plt.title(title)
for i in inds_subplots:
    if split_to_subplots:
        plt.subplot(2, 2, i)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$T_i$ [keV]')
    plt.ylabel('$Q_{fuel}$')
    plt.xlim([min(Ti_keV), max(Ti_keV)])
    plt.legend()
    plt.grid(True)
plt.tight_layout()

plt.figure(2)
title = 'Lawson Criterion:  $B=$' + str(B) + '[T], $\\beta=$' + str(plasma_beta_target) + ', $Q_{fuel}=$' + str(
    const_Q_fuel) + title_suffix
title += ', $P_{fus}$ units $[MW/m^3]$'
if split_to_subplots:
    plt.suptitle(title)
else:
    plt.title(title)
for i in inds_subplots:
    if split_to_subplots:
        plt.subplot(2, 2, i)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e21, 1e26])
    plt.xlim([min(Ti_keV), max(Ti_keV)])
    plt.xlabel('$T_i$ [keV]')
    plt.ylabel('$p \cdot \\tau$ [m$^{-3}$keV s]')
    plt.legend()
    plt.grid(True)
plt.tight_layout()

plt.figure(3)
if split_to_subplots:
    plt.suptitle(title)
else:
    plt.title(title)
for i in inds_subplots:
    if split_to_subplots:
        plt.subplot(2, 2, i)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-2, 1e3])
    plt.xlim([min(Ti_keV), max(Ti_keV)])
    plt.xlabel('$T_i$ [keV]')
    plt.ylabel('$\\tau$ [s]')
    plt.legend()
    plt.grid(True)
plt.tight_layout()

plt.figure(4)
title = 'Lawson Criterion: $B=$' + str(B) + '[T], $\\beta=$' + str(plasma_beta_target)
if split_to_subplots:
    plt.suptitle(title)
else:
    plt.title(title)
for i in inds_subplots:
    if split_to_subplots:
        plt.subplot(2, 2, i)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$T_i$ [keV]')
    plt.ylabel('$n_e[m^{-3}]$')
    plt.xlim([min(Ti_keV), max(Ti_keV)])
    plt.legend()
    plt.grid(True)
plt.tight_layout()

# ## save figs at higher res
figs_folder = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/'
# plt.figure(1)
# plt.savefig(figs_folder + 'Q_fuel_at_const_beta' + file_suffix + '.pdf', format='pdf')
# plt.figure(2)
# plt.savefig(figs_folder + 'lawson_p_tau_at_const_beta' + file_suffix + '.pdf', format='pdf')
plt.figure(3)
file_suffix = f'_B_{B}T_beta_{plasma_beta_target}'
# plt.savefig(figs_folder + 'lawson_tau_at_const_beta' + file_suffix + '.pdf', format='pdf')
# plt.figure(4)
# plt.savefig(figs_folder + 'ne_at_const_beta' + '.pdf', format='pdf')
