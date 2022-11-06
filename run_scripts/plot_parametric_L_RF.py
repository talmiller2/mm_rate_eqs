import matplotlib

# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.size': 12})

import numpy as np
from scipy.optimize import curve_fit

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from mm_rate_eqs.fusion_functions import get_lawson_parameters

from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss, get_magnetic_pressure, \
    get_ideal_gas_pressure, get_ideal_gas_energy_per_volume, get_magnetic_field_for_given_pressure, \
    get_bohm_diffusion_constant, get_larmor_radius

# plt.close('all')

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs/'

# main_dir += '/runs/slurm_runs/set41_MM_Rm_3_ni_1e21_Ti_10keV_withRF'
main_dir += '/runs/slurm_runs/set42_MM_Rm_3_ni_1e21_Ti_10keV_withRF'

# colors = ['b', 'g', 'c', 'orange', 'r', 'm']
# linestyles = ['-', '-', '-', '-', '-', '-']
# colors     = ['k', 'b', 'g', 'c', 'orange', 'r', 'm']
# linestyles = ['-', '-', '-', '-',      '-', '-', '-']
# colors     = ['k', 'b', 'g', 'c', 'orange', 'r', 'm',  'b',  'g']
# linestyles = ['-', '-', '-', '-',      '-', '-', '-', '--', '--']
# colors = ['b', 'b', 'b', 'g', 'g', 'g', 'r', 'r', 'r', 'orange', 'm', 'y']
# linestyles = ['-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '-', '-']
colors = ['b', 'b', 'b', 'g', 'g', 'r', 'r', 'k', 'k', 'k', 'y', 'y']
linestyles = ['-', '--', ':', '-', '--', '-', '--', '-', '--', ':', '--', ':']

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

linewidth = 2

RF_capacity_cl_list = []
RF_capacity_cr_list = []
RF_capacity_lc_list = []
RF_capacity_rc_list = []
set_name_list = []

# no RF
# RF_capacity_cl_list += [0]
# RF_capacity_cr_list += [0]
# RF_capacity_lc_list += [0]
# RF_capacity_rc_list += [0]

# # Rm=3, l=1m, ERF=50kV/m, alpha=1, beta=0, selectivity=1
# RF_capacity_cl_list += [0.026]
# RF_capacity_cr_list += [0.026]
# RF_capacity_lc_list += [0.61]
# RF_capacity_rc_list += [0.61]
# set_name_list += ['A']

# # # Rm=3, l=1m, ERF=50kV/m, alpha=1, beta=-1, selectivity=1.46
# RF_capacity_cl_list += [0.02]
# RF_capacity_cr_list += [0.031]
# RF_capacity_lc_list += [0.455]
# RF_capacity_rc_list += [0.664]

# Rm=3, l=1m, ERF=50kV/m, alpha=0.8, beta=-1, selectivity=2.72
# RF_capacity_cl_list += [0.024]
# RF_capacity_cr_list += [0.013]
# RF_capacity_lc_list += [0.125]
# RF_capacity_rc_list += [0.341]
# set_name_list += ['B']

# # Rm=3, l=1m, ERF=50kV/m, alpha=0.9, beta=-5, selectivity=3.31
# RF_capacity_cl_list += [0.018]
# RF_capacity_cr_list += [0.01]
# RF_capacity_lc_list += [0.089]
# RF_capacity_rc_list += [0.296]
# set_name_list += ['C']
#
# # Rm=3, l=1m, ERF=50kV/m, alpha=0.9, beta=-4, selectivity=4.46
# RF_capacity_cl_list += [0.021]
# RF_capacity_cr_list += [0.012]
# RF_capacity_lc_list += [0.1]
# RF_capacity_rc_list += [0.446]
# set_name_list += ['D']
#
# Rm=3, l=1m, ERF=50kV/m, alpha=0.9, beta=-2, selectivity=5.16
# RF_capacity_cl_list += [0.026]
# RF_capacity_cr_list += [0.014]
# RF_capacity_lc_list += [0.130]
# RF_capacity_rc_list += [0.671]
# set_name_list += ['E']
#
# # Rm=3, l=1m, ERF=50kV/m, alpha=0.8, beta=-5, selectivity=6.52
# RF_capacity_cl_list += [0.023]
# RF_capacity_cr_list += [0.008]
# RF_capacity_lc_list += [0.074]
# RF_capacity_rc_list += [0.484]
# set_name_list += ['F']

# # Rm=3, l=1m, ERF=50kV/m, alpha=0.8, beta=-4, selectivity=7.46
# RF_capacity_cl_list += [0.027]
# RF_capacity_cr_list += [0.009]
# RF_capacity_lc_list += [0.079]
# RF_capacity_rc_list += [0.593]


for ind_RF in range(len(RF_capacity_cl_list)):
    color = colors[ind_RF]
    linestyle = linestyles[ind_RF]
    plasma_mode = 'isoT'

    flux_list = np.nan * np.zeros(len(num_cells_list))
    n1_list = np.nan * np.zeros(len(num_cells_list))
    for ind_N, number_of_cells in enumerate(num_cells_list):
        run_name = plasma_mode
        RF_label = 'RF_terms_' + 'cl_' + str(RF_capacity_cl_list[ind_RF]) \
                   + '_cr_' + str(RF_capacity_cr_list[ind_RF]) \
                   + '_lc_' + str(RF_capacity_lc_list[ind_RF]) \
                   + '_rc_' + str(RF_capacity_rc_list[ind_RF])
        run_name += '_' + RF_label
        run_name += '_N_' + str(number_of_cells)

        # if ind_N == 0:
        #     print('run_name = ' + run_name)

        save_dir = main_dir + '/' + run_name

        state_file = save_dir + '/state.pickle'
        settings_file = save_dir + '/settings.pickle'

        try:
            state, settings = load_simulation(state_file, settings_file)

            # post process the flux normalization
            # norm_factor = 2.0 * settings['cross_section_main_cell'] * settings['transmission_factor']
            # norm_factor = 2.0 * settings['cross_section_main_cell']
            # norm_factor *= state['n'][0] * state['v_th'][0]
            # norm_factor = state['n'][0] * state['v_th'][0]
            # state['flux_mean'] /= norm_factor
            ni = state['n'][0]
            Ti_keV = state['Ti'][0] / 1e3
            _, flux_lawson = get_lawson_parameters(ni, Ti_keV, settings)
            state['flux_mean'] *= settings['cross_section_main_cell']
            state['flux_mean'] /= flux_lawson

            flux_list[ind_N] = state['flux_mean']
            # n1_list[ind_N] = state['n'][-1]
            # n1_list[ind_N] = state['n'][-2]

            selected_number_of_cells = 30
            if number_of_cells == selected_number_of_cells:
                ni_save = state['n']

        except:
            pass

    if RF_capacity_lc_list[ind_RF] > 0:
        selectivity = '{:.2f}'.format(RF_capacity_rc_list[ind_RF] / RF_capacity_lc_list[ind_RF])
        selectivity_trapped = '{:.2f}'.format(RF_capacity_cr_list[ind_RF] / RF_capacity_cl_list[ind_RF])
    else:
        selectivity = '1'
    label = ''
    label += '$' + 's=' + selectivity + '$'
    # label += ', $' + 's_t=' + selectivity_trapped + '$'
    label += ', $ \\bar{N}_{cl}=' + str(RF_capacity_cl_list[ind_RF]) \
             + ', \\bar{N}_{cr}=' + str(RF_capacity_cr_list[ind_RF]) \
             + ', \\bar{N}_{lc}=' + str(RF_capacity_lc_list[ind_RF]) \
             + ', \\bar{N}_{rc}=' + str(RF_capacity_rc_list[ind_RF]) + '$'
    # label = 'set ' + set_name_list[ind_RF]
    print(label)

    # plot flux as a function of N
    plt.figure(1)
    plt.plot(num_cells_list, flux_list, label=label, linestyle=linestyle, color=color, linewidth=linewidth)

    # extract the density profile
    plt.figure(2)
    x = np.linspace(0, selected_number_of_cells, selected_number_of_cells)
    plt.plot(x, ni_save, '-', label=label, linestyle=linestyle, color=color, linewidth=linewidth)


# add plot for then radial flux in the MM section alone

B = 3.0  # T
# B = 10.0  # T
D_bohm = get_bohm_diffusion_constant(state['Te'][0], B)  # [m^2/s]
# integral of dn/dz for linearly declining n is n*L/2
dn_dr = state['n'][0] * np.ones(len(num_cells_list)) / 2 / (settings['diameter_main_cell'] / 2)
radial_flux_density = D_bohm * dn_dr
system_total_length = np.array(num_cells_list) * settings['cell_size']
cyllinder_radial_cross_section = np.pi * settings['diameter_main_cell'] * system_total_length
radial_flux_bohm = radial_flux_density * cyllinder_radial_cross_section
radial_flux_bohm /= flux_lawson

gyro_radius = get_larmor_radius(state['Ti'][0], B)
D_classical = gyro_radius ** 2 * state['coulomb_scattering_rate'][0]
dn_dr = state['n'][0] * np.ones(len(num_cells_list)) / 3 / (settings['diameter_main_cell'] / 2)
radial_flux_density = D_classical * dn_dr
radial_flux_classical = radial_flux_density * cyllinder_radial_cross_section
radial_flux_classical /= flux_lawson

# fig = plt.figure(1)
# plt.plot(num_cells_list, radial_flux_bohm, label='radial bohm', linestyle='-', color='k', linewidth=linewidth)
# plt.plot(num_cells_list, radial_flux_classical, label='radial classical', linestyle='--', color='k', linewidth=linewidth)

fig = plt.figure(1)
plt.yscale("log")
# plt.xscale("log")
plt.xlabel('N')
# plt.ylabel('flux [$s^{-1}$]')
# plt.ylabel('$\\phi_{p}$ [$m^{-2}s^{-1}$]')
# plt.ylabel('$\\phi_{p} / \\phi_{p,0}$')
plt.ylabel('$\\phi_{ss} / \\phi_{Lawson}$')
# plt.title('flux as a function of system size')
# plt.title('flux as a function of system size ($U/v_{th}$=' + str(U) + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()

# text = '(a)'
# # text = '(b)'
# plt.text(0.98, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
#          horizontalalignment='right', verticalalignment='top',
#          transform=fig.axes[0].transAxes)
#
# ax = plt.gca()
# ax.set_xticks([10, 100])
# ax.set_yticks([10, 100])
# # ax.set_yticks([1000, 2000, 4000])
# from matplotlib.ticker import StrMethodFormatter, NullFormatter
#
# ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
# ax.xaxis.set_minor_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
# ax.yaxis.set_minor_formatter(NullFormatter())

plt.figure(2)
plt.xlabel('cell number')
# plt.xlabel('N')
plt.ylabel('ion density [$m^{-3}$]')
# plt.ylabel('$n_1$ [$m^{-3}$]')
# plt.title('density profile (N=' + str(chosen_num_cells) + ')')
# plt.title('density profile (N=' + str(chosen_num_cells) + ' cells, $U/v_{th}$=' + str(U) + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()

# save pics in high res
# save_dir = '../../../Papers/texts/paper2020/pics/'
save_dir = '../../../Papers/texts/paper2022/pics/'
# save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics/'
# save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics_with_Rm_10/'

# file_name = 'flux_function_of_N'
# file_name = 'flux_function_of_N_suboptimal'
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
#
# file_name = 'n_function_of_cell_number'
# beingsaved = plt.figure(2)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
