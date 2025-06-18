import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 12
plt.close('all')

from mm_rate_eqs.fusion_functions import load_sigma_v_fusion_files, get_fusion_power_multiple_ions, \
    set_ion_densities_quasi_neutral, update_ion_latex_name

from mm_rate_eqs.plasma_functions import get_brem_radiation_loss_relativistic, get_cyclotron_radiation_loss, \
    get_cyclotron_radiation_loss_envelope

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
linestyle_list = ['-', '-', '-', '-', '--', ':', '-']

process_opt_dict = {}

for ip, process in enumerate(process_list):
    color = color_list[ip]
    linestyle = linestyle_list[ip]

    print('*****', process, '*****')

    # normalize ion densities to satisfy quasi-neutrality with electrons
    # ne = 1e20  # [m^-3]
    ne = 1e21  # [m^-3]
    ions_list, Zi_list, ni_array, ni = set_ion_densities_quasi_neutral(process, ne, Ti_keV, sigma_v_dict)

    # # plot relative ion densities
    # plt.figure()
    # for ni_curr, ion, color in zip(ni_array, ions_list, ['b', 'g', 'r']):
    #     plt.plot(Ti_keV, ni_curr / ni * 100, label=ion, color=color)
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
            plt.plot(Ti_keV, ni_curr, label=update_ion_latex_name(ion), color=color)
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

    P_fus_tot, P_fus_charged_tot = get_fusion_power_multiple_ions(ni_array, Ti_keV, ions_list, sigma_v_dict)

    plt.plot(Ti_keV, P_fus_tot, '-k', linewidth=3, label='fusion total')
    plt.plot(Ti_keV, P_fus_charged_tot, '-r', linewidth=2, label='fusion charged')

    a = 1
    R = 2
    r = 0

    Te_keV = 1.0 * Ti_keV
    Te_suffix_1 = ' $T_e=T_i$'
    B = 10  # [T]
    B_suffix_1 = ', $B=10$[T]'
    # B = 20  # [T]
    # B_suffix_1 = ', $B=20$[T]'
    P_brem = get_brem_radiation_loss_relativistic(ni_array, Zi_list, Te_keV, use_relativistic_correction=True)
    P_cyc = get_cyclotron_radiation_loss(ne, Te_keV, B, version='Stacey')
    P_cyc_Kukushkin = get_cyclotron_radiation_loss(ne, Te_keV, B, version='Kukushkin', a=a, R=R, r=r)
    P_cyc_Trubnikov = get_cyclotron_radiation_loss(ne, Te_keV, B, version='Trubnikov', a=a, R=R, r=r)
    P_cyc_Wiedemann = get_cyclotron_radiation_loss(ne, Te_keV, B, version='Wiedemann', a=a, R=R, r=r)
    P_cyc_min, P_cyc_max = get_cyclotron_radiation_loss_envelope(ne, Te_keV, B, a=a, R=R, r=r)
    # Te_keV = 0.1 * Ti_keV
    # Te_suffix_2 = ' $T_e=T_i /10$'
    Te_keV = 0.4 * Ti_keV
    Te_suffix_2 = ' $T_e=0.4 T_i$'
    # B = 10  # [T]
    # B_suffix_2 = ', $B=10$[T]'
    B = 20  # [T]
    B_suffix_2 = ', $B=20$[T]'
    P_brem_2 = get_brem_radiation_loss_relativistic(ni_array, Zi_list, Te_keV, use_relativistic_correction=False)
    P_cyc_2 = get_cyclotron_radiation_loss(ne, Te_keV, B)
    P_cyc_min_2, P_cyc_max_2 = get_cyclotron_radiation_loss_envelope(ne, Te_keV, B, a=a, R=R, r=r)

    plt.plot(Ti_keV, P_brem, linestyle='-', color='b', label='brem' + Te_suffix_1)
    # plt.plot(Ti_keV, P_brem_2, linestyle='--', color='b', label='brem' + Te_suffix_2)
    plt.plot(Ti_keV, P_cyc, linestyle='-', color='g', label='cyc Stacey' + Te_suffix_1 + B_suffix_1)
    # plt.plot(Ti_keV, P_cyc_2, linestyle='--', color='g', label='cyc Stacey' + Te_suffix_2 + B_suffix_2)
    # plt.plot(Ti_keV, P_cyc_Kukushkin, linestyle='-', color='teal', label='cyc Kukushkin' + Te_suffix_1 + B_suffix_1)
    # plt.plot(Ti_keV, P_cyc_Trubnikov, linestyle='-', color='magenta', label='cyc Trubnikov' + Te_suffix_1 + B_suffix_1)
    # plt.plot(Ti_keV, P_cyc_Wiedemann, linestyle='-', color='pink', label='cyc Wiedemann' + Te_suffix_1 + B_suffix_1)
    plt.fill_between(Ti_keV, P_cyc_min, P_cyc_max, color='g', alpha=0.3, label='cyc' + Te_suffix_1 + B_suffix_1)
    # plt.fill_between(Ti_keV, P_cyc_min_2, P_cyc_max_2, color='orange', alpha=0.3, label='cyc' + Te_suffix_2 + B_suffix_2)

    plt.suptitle('fusion fuel: ' + update_ion_latex_name(process))
    # plt.title(process)
    plt.title('Power density')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$T_i$ [keV]')
    plt.ylabel('[W/m$^3$]')
    plt.xlim([min(Ti_keV), max(Ti_keV)])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ## save fig at higher res
    # figs_folder = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/'
    # process_underscore = process.replace(' ', '_')
    # plt.savefig(figs_folder + 'power_' + process_underscore + '.pdf', format='pdf')

# plt.figure()
# T =  np.linspace(0, 1000, 1000)
# t = T / 511
# y1 = (1 + 1.78 * t ** 1.34)
# y2 = 2.12 * t * (1 + 1.1 * t + t ** 2.0 - 1.25 * t ** 2.5)
# # plt.plot(T, y1, label='$P_{brem}$ relativistic factor')
# # plt.plot(T, y2, label='$P_{brem}$ relativistic addition')
# for Z_eff in [1, 2, 3]:
#     y = Z_eff * y1 + y2
#     plt.plot(T, y / Z_eff, label='$Z_{eff}$=' + str(Z_eff))
# plt.title('brem relativistic correction $P_{rel}/P_{norel}$')
# plt.xlabel('$T_e$ [keV]')
# plt.legend()
# plt.grid(True)
