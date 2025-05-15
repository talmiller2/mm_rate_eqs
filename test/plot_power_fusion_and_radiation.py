import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 12
plt.close('all')

from mm_rate_eqs.fusion_functions import load_sigma_v_fusion_files, get_E_reaction, get_E_charged
from mm_rate_eqs.plasma_functions import get_brem_radiation_loss_relativistic, get_cyclotron_radiation_loss, \
    define_electron_charge, get_brem_radiation_loss

# Ti_keV = np.linspace(1, 300, 1000)
Ti_keV = np.linspace(1, 1000, 1000)
# Ti_keV = np.linspace(1, 200, 1000)

# sigma_v_dict = {}
# reactions = ['D_T_to_n_alpha', 'D_He3_to_p_alpha', 'D_D_to_p_T', 'D_D_to_n_He3', 'p_B_to_3alpha']
# for reaction in reactions:
#     sigma_v_dict[reaction] = get_sigma_v_fusion_sampled(Ti_keV, reaction=reaction)
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
linestyle_list = ['-', '-', '-', '-', '--', ':', '-']

process_opt_dict = {}

for ip, process in enumerate(process_list):
    color = color_list[ip]
    linestyle = linestyle_list[ip]

    print('*****', process, '*****')

    if process == 'D-T':
        ions_list = ['D', 'T']
        Zi_list = [1, 1]
        # for const ne, optimal choice to maximize P_fus is (1/2Z), proof in Wurzel2022
        ni_rel_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zi_list]

    elif process == 'D-He3':
        ions_list = ['D', 'He3']
        Zi_list = [1, 2]
        ni_rel_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zi_list]

    elif process == 'p-B11':
        ions_list = ['p', 'B11']
        Zi_list = [1, 5]
        ni_rel_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zi_list]

    elif process == 'pure D-D':
        ions_list = ['D']
        Zi_list = [1]
        ni_rel_list = [1 + 0 * Ti_keV]

    elif process == 'fully-catalyzed D-D':
        # from Wurzel2022: "Here, we only consider the steady-state reaction path where He3 and T react with D at the
        # same rate as they are created in each branch of the D-D reaction. Furthermore, we assume an idealized scenario
        # without synchrotron radiation and that the “ash” alpha particles and protons immediately exit after depositing
        # their energy and comprise a negligible fraction of ions in the plasma. Finally, we assume that D is added at
        # the same rate as it is consumed." See Eqs. (C13)-(C14).
        ions_list = ['D', 'T', 'He3']
        Zi_list = [1, 1, 2]
        ni_rel_list = [1 + 0 * Ti_keV,
                       0.5 * sigma_v_dict['D_D_to_p_T'] / sigma_v_dict['D_T_to_n_alpha'],
                       0.5 * sigma_v_dict['D_D_to_n_He3'] / sigma_v_dict['D_He3_to_p_alpha']]


    elif process == 'He3-catalyzed D-D':
        ions_list = ['D', 'He3']
        Zi_list = [1, 2]
        # densities chosen at steady state for D,He3 (T assumed as extracted instantly)
        ni_rel_list = [1 + 0 * Ti_keV,
                       0.5 * sigma_v_dict['D_D_to_n_He3'] / sigma_v_dict['D_He3_to_p_alpha']]

    elif process == 'T-catalyzed D-D':
        ions_list = ['D', 'T']
        Zi_list = [1, 1]
        # densities chosen at steady state for D,T (He3 assumed as extracted instantly)
        ni_rel_list = [1 + 0 * Ti_keV,
                       0.5 * sigma_v_dict['D_D_to_p_T'] / sigma_v_dict['D_T_to_n_alpha']]

    else:
        raise ValueError('invalid process', process)

    # normalize ion densities to satisfy quasi-neutrality with electrons
    ne = 1e20  # [m^-3]
    # ne = 1e21 # [m^-3]
    ni_rel_array = np.array(ni_rel_list)
    ni_rel_sum = np.sum(ni_rel_array, axis=0)
    neutrality_fac = sum([ni_rel * Zi for ni_rel, Zi in zip(ni_rel_array, Zi_list)])
    ni_array = 0 * ni_rel_array
    for ind_ion in range(ni_rel_array.shape[0]):
        ni_array[ind_ion, :] = ni_rel_array[ind_ion, :] * ne / neutrality_fac
    ni = np.sum(ni_array, axis=0)

    # # plot ni_rel
    # plt.figure()
    # for ni_rel, ion, color in zip(ni_rel_list, ions_list, ['b', 'g', 'r']):
    #     plt.plot(Ti_keV, ni_rel / ni_rel_sum * 100, label=ion, color=color)
    # plt.title(process)
    # plt.xscale('log')
    # plt.xlabel('$T_i$ [keV]')
    # plt.ylabel('$n_i$ [%]')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # plot densities
    # plt.figure()
    if 'catalyzed' in process:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(Ti_keV, ne + 0 * Ti_keV, '--k', label='electrons')
        plt.plot(Ti_keV, ni, '-k', label='ions')
        for ni_curr, ion, color in zip(ni_array, ions_list, ['b', 'g', 'r']):
            plt.plot(Ti_keV, ni_curr, label=ion, color=color)
        # plt.title(process)
        plt.title('Particle density')
        plt.xscale('log')
        plt.xlabel('$T_i$ [keV]')
        plt.ylabel('$n$ [m$^{-3}$]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    else:
        pass

    # plot power
    if 'catalyzed' in process:
        plt.subplot(1, 2, 2)
    else:
        plt.figure(figsize=(7, 6))
        plt.subplot(1, 1, 1)
    MeV_to_J = define_electron_charge() * 1e6
    P_fus_tot = 0 * Ti_keV
    P_fus_charged_tot = 0 * Ti_keV
    for ind_r_1, reactant_1 in enumerate(ions_list):
        for ind_r_2, reactant_2 in enumerate(ions_list):
            # print(reactant_1, reactant_2)
            # by default some processes are neglected (small or no available data)
            if (reactant_1 == 'D' and reactant_2 == 'T') or (reactant_1 == 'T' and reactant_2 == 'D'):
                reactions = ['D_T_to_n_alpha']
            elif reactant_1 == 'D' and reactant_2 == 'D':
                reactions = ['D_D_to_p_T', 'D_D_to_n_He3']
            elif (reactant_1 == 'D' and reactant_2 == 'He3') or (reactant_1 == 'He3' and reactant_2 == 'D'):
                reactions = ['D_He3_to_p_alpha']
            elif reactant_1 == 'p' and reactant_2 == 'B11':
                reactions = ['p_B_to_3alpha']
            else:
                reactions = []

            for reaction in reactions:
                sigma_v_curr = sigma_v_dict[reaction]
                if reactant_1 == reactant_2:
                    sigma_v_curr /= 2.0  # division by (1 + delta_jk)
                ni_1_curr = ni_array[ind_r_1, :]
                ni_2_curr = ni_array[ind_r_2, :]
                E_f_curr = get_E_reaction(reaction=reaction)  # [MeV]
                E_ch_curr = get_E_charged(reaction=reaction)  # [MeV]
                P_fus_curr = ni_1_curr * ni_2_curr * sigma_v_curr * E_f_curr * MeV_to_J
                P_fus_charged_curr = ni_1_curr * ni_2_curr * sigma_v_curr * E_ch_curr * MeV_to_J
                P_fus_tot += P_fus_curr
                P_fus_charged_tot += P_fus_charged_curr
                # plt.plot(Ti_keV, P_fus_curr, linestyle=':', label='fus: ' + reactant_1 + ' + ' + reactant_2)
                # plt.plot(Ti_keV, P_fus_charged_curr, linestyle=':', label='fus charged: ' + reactant_1 + ' + ' + reactant_2)

    plt.plot(Ti_keV, P_fus_tot, '-k', linewidth=3, label='fusion total')
    plt.plot(Ti_keV, P_fus_charged_tot, '-r', linewidth=2, label='fusion charged')

    B = 10  # [T]
    Te_keV = 1.0 * Ti_keV
    Te_suffix_1 = ' $T_e=T_i$'
    B_suffix_1 = ', $B=10$[T]'
    P_brem = get_brem_radiation_loss_relativistic(ni_array, Zi_list, Te_keV)
    # P_brem_old = get_brem_radiation_loss(ne + 0 * Ti_keV, ne + 0 * Ti_keV, Te_keV, 1)
    P_cyc = get_cyclotron_radiation_loss(ne, Te_keV, B)
    B = 10  # [T]
    Te_keV = 0.1 * Ti_keV
    Te_suffix_2 = ' $T_e=T_i /10$'
    B_suffix_2 = ', $B=10$[T]'
    P_brem_2 = get_brem_radiation_loss_relativistic(ni_array, Zi_list, Te_keV)
    P_cyc_2 = get_cyclotron_radiation_loss(ne, Te_keV, B)
    B = 1  # [T]
    Te_keV = 1.0 * Ti_keV
    Te_suffix_3 = ' $T_e=T_i$'
    B_suffix_3 = ', $B=1$[T]'
    P_cyc_3 = get_cyclotron_radiation_loss(ne, Te_keV, B)
    plt.plot(Ti_keV, P_brem, linestyle='-', color='b', label='brem' + Te_suffix_1)
    # plt.plot(Ti_keV, P_brem_old, linestyle=':', color='b', label='brem $T_e=T_i$ old')
    plt.plot(Ti_keV, P_brem_2, linestyle='--', color='b', label='brem' + Te_suffix_2)
    plt.plot(Ti_keV, P_cyc, linestyle='-', color='g', label='cyc' + Te_suffix_1 + B_suffix_1)
    plt.plot(Ti_keV, P_cyc_2, linestyle='--', color='g', label='cyc' + Te_suffix_2 + B_suffix_2)
    plt.plot(Ti_keV, P_cyc_3, linestyle=':', color='g', label='cyc' + Te_suffix_3 + B_suffix_3)

    plt.suptitle('fusion fuel: ' + process)
    # plt.title(process)
    plt.title('Power density')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$T_i$ [keV]')
    plt.ylabel('[W/m$^3$]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # # save fig at higher res
    # process_underscore = process.replace(' ', '_')
    # plt.savefig('/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/power_' + process_underscore + '.png', format='png', dpi=600)
