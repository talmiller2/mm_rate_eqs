import matplotlib
# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

import matplotlib.pyplot as plt
from matplotlib import cm

# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.size': 12})

import numpy as np
from scipy.optimize import curve_fit

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from mm_rate_eqs.fusion_functions import get_lawson_parameters, get_lawson_criterion_piel

from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss, get_magnetic_pressure, \
    get_ideal_gas_pressure, get_ideal_gas_energy_per_volume, get_magnetic_field_for_given_pressure, \
    get_bohm_diffusion_constant, get_larmor_radius

plt.close('all')

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs/'

# main_dir += '/runs/slurm_runs/set41_MM_Rm_3_ni_1e21_Ti_10keV_withRF'
main_dir += '/runs/slurm_runs/set42_MM_Rm_3_ni_1e21_Ti_10keV_withRF'

RF_type = 'electric_transverse'
# RF_type = 'magnetic_transverse'

# colors = ['b', 'g', 'c', 'orange', 'r', 'm']
# linestyles = ['-', '-', '-', '-', '-', '-']
# colors     = ['k', 'b', 'g', 'c', 'orange', 'r', 'm']
# linestyles = ['-', '-', '-', '-',      '-', '-', '-']
# colors     = ['k', 'b', 'g', 'c', 'orange', 'r', 'm',  'b',  'g']
# linestyles = ['-', '-', '-', '-',      '-', '-', '-', '--', '--']
# colors = ['b', 'b', 'b', 'g', 'g', 'g', 'r', 'r', 'r', 'orange', 'm', 'y']
# linestyles = ['-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '-', '-']
# colors     = ['b',  'b', 'b', 'g', 'g', 'r', 'r', 'k', 'k', 'k', 'y', 'y']
# linestyles = ['-', '--', ':', '-', '--', '-', '--', '-', '--', ':', '--', ':']
colors = ['b', 'b', 'k', 'k', 'g', 'g', 'r', 'r', 'm', 'm', 'k', 'k']
linestyles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--']

# num_sets = 9
num_sets = 4
# colors = []
# linestyles = []
# for i in range(num_sets):
#     colors += [cm.rainbow(1.0 * i / num_sets)]
#     colors += [cm.rainbow(1.0 * i / num_sets)]
#     linestyles += ['-', '--']

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# linewidth = 1
linewidth = 2

# RF_cl_list = []
# RF_cr_list = []
# RF_lc_list = []
# RF_rc_list = []
# set_name_list = []

# no RF
# RF_cl_list += [0]
# RF_cr_list += [0]
# RF_lc_list += [0]
# RF_rc_list += [0]

# # Rm=3, l=1m, ERF=50kV/m, alpha=1, beta=0, selectivity=1
# RF_cl_list += [0.026]
# RF_cr_list += [0.026]
# RF_lc_list += [0.61]
# RF_rc_list += [0.61]
# set_name_list += ['A']

# # # Rm=3, l=1m, ERF=50kV/m, alpha=1, beta=-1, selectivity=1.46
# RF_cl_list += [0.02]
# RF_cr_list += [0.031]
# RF_lc_list += [0.455]
# RF_rc_list += [0.664]

# Rm=3, l=1m, ERF=50kV/m, alpha=0.8, beta=-1, selectivity=2.72
# RF_cl_list += [0.024]
# RF_cr_list += [0.013]
# RF_lc_list += [0.125]
# RF_rc_list += [0.341]
# set_name_list += ['B']

# # Rm=3, l=1m, ERF=50kV/m, alpha=0.9, beta=-5, selectivity=3.31
# RF_cl_list += [0.018]
# RF_cr_list += [0.01]
# RF_lc_list += [0.089]
# RF_rc_list += [0.296]
# set_name_list += ['C']
#
# # Rm=3, l=1m, ERF=50kV/m, alpha=0.9, beta=-4, selectivity=4.46
# RF_cl_list += [0.021]
# RF_cr_list += [0.012]
# RF_lc_list += [0.1]
# RF_rc_list += [0.446]
# set_name_list += ['D']
#
# Rm=3, l=1m, ERF=50kV/m, alpha=0.9, beta=-2, selectivity=5.16
# RF_cl_list += [0.026]
# RF_cr_list += [0.014]
# RF_lc_list += [0.130]
# RF_rc_list += [0.671]
# set_name_list += ['E']
#
# # Rm=3, l=1m, ERF=50kV/m, alpha=0.8, beta=-5, selectivity=6.52
# RF_cl_list += [0.023]
# RF_cr_list += [0.008]
# RF_lc_list += [0.074]
# RF_rc_list += [0.484]
# set_name_list += ['F']

# # Rm=3, l=1m, ERF=50kV/m, alpha=0.8, beta=-4, selectivity=7.46
# RF_cl_list += [0.027]
# RF_cr_list += [0.009]
# RF_lc_list += [0.079]
# RF_rc_list += [0.593]


###########################
###########################

set_name_list = []
gas_type_list = []
RF_rc_list = []
RF_lc_list = []
RF_cr_list = []
RF_cl_list = []

###########################
###########################

# # set1, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=1.679, k/2pi=-3
# set_num = 1
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.311]
# RF_lc_list += [0.388]
# RF_cr_list += [0.025]
# RF_cl_list += [0.021]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.629]
# RF_lc_list += [0.165]
# RF_cr_list += [0.017]
# RF_cl_list += [0.025]
#
# # # set2, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=1.559, k/2pi=0
# # set_num = 2
# # set_name_list += [str(set_num) + ' (D)']
# # gas_type_list += ['deuterium']
# # RF_rc_list += [0.832]
# # RF_lc_list += [0.805]
# # RF_cr_list += [0.018]
# # RF_cl_list += [0.015]
# # set_name_list += [str(set_num) + ' (T)']
# # gas_type_list += ['tritium']
# # RF_rc_list += [0.300]
# # RF_lc_list += [0.299]
# # RF_cr_list += [0.023]
# # RF_cl_list += [0.020]
#
# # set3, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=1.199, k/2pi=-3
# # set_num = 3
# set_num = 2
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.640]
# RF_lc_list += [0.122]
# RF_cr_list += [0.014]
# RF_cl_list += [0.020]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.376]
# RF_lc_list += [0.401]
# RF_cr_list += [0.026]
# RF_cl_list += [0.023]
#
# # # set4, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=0.720, k/2pi=-2
# # set_num = 4
# # set_name_list += [str(set_num) + ' (D)']
# # gas_type_list += ['deuterium']
# # RF_rc_list += [0.297]
# # RF_lc_list += [0.081]
# # RF_cr_list += [0.017]
# # RF_cl_list += [0.010]
# # set_name_list += [str(set_num) + ' (T)']
# # gas_type_list += ['tritium']
# # RF_rc_list += [0.818]
# # RF_lc_list += [0.131]
# # RF_cr_list += [0.024]
# # RF_cl_list += [0.027]
#
# # set5, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=0.839, k/2pi=-3.0
# # set_num = 5
# set_num = 3
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.682]
# RF_lc_list += [0.076]
# RF_cr_list += [0.019]
# RF_cl_list += [0.022]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.587]
# RF_lc_list += [0.139]
# RF_cr_list += [0.015]
# RF_cl_list += [0.026]
#
# # # set6, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=0.720, k/2pi= -4.0
# # set_num = 6
# # set_name_list += [str(set_num) + ' (D)']
# # gas_type_list += ['deuterium']
# # RF_rc_list += [0.629]
# # RF_lc_list += [0.066]
# # RF_cr_list += [0.016]
# # RF_cl_list += [0.018]
# # set_name_list += [str(set_num) + ' (T)']
# # gas_type_list += ['tritium']
# # RF_rc_list += [0.555]
# # RF_lc_list += [0.101]
# # RF_cr_list += [0.015]
# # RF_cl_list += [0.020]
# #
# # # set7, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=0.660, k/2pi=-3.0
# # set_num = 7
# # set_name_list += [str(set_num) + ' (D)']
# # gas_type_list += ['deuterium']
# # RF_rc_list += [0.456]
# # RF_lc_list += [0.073]
# # RF_cr_list += [0.018]
# # RF_cl_list += [0.012]
# # set_name_list += [str(set_num) + ' (T)']
# # gas_type_list += ['tritium']
# # RF_rc_list += [0.720]
# # RF_lc_list += [0.103]
# # RF_cr_list += [0.027]
# # RF_cl_list += [0.026]
# #
# # # set8, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=0.600, k/2pi=-4
# # set_num = 8
# # set_name_list += [str(set_num) + ' (D)']
# # gas_type_list += ['deuterium']
# # RF_rc_list += [0.561]
# # RF_lc_list += [0.056]
# # RF_cr_list += [0.016]
# # RF_cl_list += [0.016]
# # set_name_list += [str(set_num) + ' (T)']
# # gas_type_list += ['tritium']
# # RF_rc_list += [0.632]
# # RF_lc_list += [0.086]
# # RF_cr_list += [0.021]
# # RF_cl_list += [0.020]
#
# # set9, Rm=3, l=1m, ERF=50kV/m, omega/omega0T=0.660, k/2pi=-7
# # set_num = 9
# set_num = 4
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.504]
# RF_lc_list += [0.047]
# RF_cr_list += [0.020]
# RF_cl_list += [0.016]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.361]
# RF_lc_list += [0.058]
# RF_cr_list += [0.013]
# RF_cl_list += [0.018]

#########################


# # set1, Rm=3, l=1m, BRF=0.04T, omega/omega0T=1.679, k/2pi=-3
# set_num = 1
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.240]
# RF_lc_list += [0.319]
# RF_cr_list += [0.038]
# RF_cl_list += [0.058]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.368]
# RF_lc_list += [0.195]
# RF_cr_list += [0.039]
# RF_cl_list += [0.052]
#
# # set2, Rm=3, l=1m, BRF=0.04T, omega/omega0T=1.559, k/2pi=0
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.485]
# RF_lc_list += [0.469]
# RF_cr_list += [0.132]
# RF_cl_list += [0.142]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.274]
# RF_lc_list += [0.276]
# RF_cr_list += [0.036]
# RF_cl_list += [0.030]
#
# # set3, Rm=3, l=1m, BRF=0.04T, omega/omega0T=1.199, k/2pi=-3
# set_num = 3
# # set_num = 2
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.573]
# RF_lc_list += [0.096]
# RF_cr_list += [0.052]
# RF_cl_list += [0.061]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.286]
# RF_lc_list += [0.198]
# RF_cr_list += [0.042]
# RF_cl_list += [0.043]
#
# # set4, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.720, k/2pi=-2
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.210]
# RF_lc_list += [0.066]
# RF_cr_list += [0.014]
# RF_cl_list += [0.013]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.345]
# RF_lc_list += [0.094]
# RF_cr_list += [0.022]
# RF_cl_list += [0.031]
#
# # set5, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.839, k/2pi=-3.0
# set_num = 5
# # set_num = 3
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.342]
# RF_lc_list += [0.065]
# RF_cr_list += [0.021]
# RF_cl_list += [0.029]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.452]
# RF_lc_list += [0.101]
# RF_cr_list += [0.035]
# RF_cl_list += [0.047]
#
# # set6, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.720, k/2pi= -4.0
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.331]
# RF_lc_list += [0.049]
# RF_cr_list += [0.023]
# RF_cl_list += [0.024]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.397]
# RF_lc_list += [0.081]
# RF_cr_list += [0.029]
# RF_cl_list += [0.036]
#
# # set7, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.660, k/2pi=-3.0
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.268]
# RF_lc_list += [0.048]
# RF_cr_list += [0.016]
# RF_cl_list += [0.017]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.431]
# RF_lc_list += [0.087]
# RF_cr_list += [0.025]
# RF_cl_list += [0.031]
#
# # set8, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.600, k/2pi=-4
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.322]
# RF_lc_list += [0.053]
# RF_cr_list += [0.019]
# RF_cl_list += [0.023]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.383]
# RF_lc_list += [0.074]
# RF_cr_list += [0.023]
# RF_cl_list += [0.036]
#
# # set9, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.660, k/2pi=-7
# set_num = 9
# # set_num = 4
# set_name_list += [str(set_num) + ' (D)']
# gas_type_list += ['deuterium']
# RF_rc_list += [0.437]
# RF_lc_list += [0.041]
# RF_cr_list += [0.029]
# RF_cl_list += [0.038]
# set_name_list += [str(set_num) + ' (T)']
# gas_type_list += ['tritium']
# RF_rc_list += [0.354]
# RF_lc_list += [0.058]
# RF_cr_list += [0.031]
# RF_cl_list += [0.019]

#########################


# set1, Rm=3, l=1m, ERF=25kV/m, omega/omega0=1.679 k/2pi=3.0
set_num = 1
set_name_list += [str(set_num) + ' (D)']
gas_type_list += ['deuterium']
RF_rc_list += [0.223]
RF_lc_list += [0.211]
RF_cr_list += [0.024]
RF_cl_list += [0.017]
set_name_list += [str(set_num) + ' (T)']
gas_type_list += ['tritium']
RF_rc_list += [0.356]
RF_lc_list += [0.086]
RF_cr_list += [0.014]
RF_cl_list += [0.020]

# set2, Rm=3, l=1m, ERF=25kV/m, omega/omega0=1.199 k/2pi=-3.0
set_num = 2
set_name_list += [str(set_num) + ' (D)']
gas_type_list += ['deuterium']
RF_rc_list += [0.440]
RF_lc_list += [0.053]
RF_cr_list += [0.023]
RF_cl_list += [0.024]
set_name_list += [str(set_num) + ' (T)']
gas_type_list += ['tritium']
RF_rc_list += [0.185]
RF_lc_list += [0.270]
RF_cr_list += [0.021]
RF_cl_list += [0.021]

# set3, Rm=3, l=1m, ERF=25kV/m, omega/omega0=0.839 k/2pi=-3.0
set_num = 3
set_name_list += [str(set_num) + ' (D)']
gas_type_list += ['deuterium']
RF_rc_list += [0.371]
RF_lc_list += [0.032]
RF_cr_list += [0.012]
RF_cl_list += [0.015]
set_name_list += [str(set_num) + ' (T)']
gas_type_list += ['tritium']
RF_rc_list += [0.377]
RF_lc_list += [0.067]
RF_cr_list += [0.027]
RF_cl_list += [0.017]

# set4, Rm=3, l=1m, ERF=25kV/m, omega/omega0=0.660 k/2pi=-7.0
set_num = 4
set_name_list += [str(set_num) + ' (D)']
gas_type_list += ['deuterium']
RF_rc_list += [0.298]
RF_lc_list += [0.025]
RF_cr_list += [0.014]
RF_cl_list += [0.014]
set_name_list += [str(set_num) + ' (T)']
gas_type_list += ['tritium']
RF_rc_list += [0.228]
RF_lc_list += [0.033]
RF_cr_list += [0.014]
RF_cl_list += [0.008]

for ind_RF in range(len(RF_cl_list)):
    color = colors[ind_RF]
    linestyle = linestyles[ind_RF]
    plasma_mode = 'isoT'

    flux_list = np.nan * np.zeros(len(num_cells_list))
    n1_list = np.nan * np.zeros(len(num_cells_list))
    for ind_N, number_of_cells in enumerate(num_cells_list):
        run_name = plasma_mode
        run_name += '_' + gas_type_list[ind_RF]
        RF_label = 'RF_terms_' \
                   + 'cl_' + str(RF_cl_list[ind_RF]) \
                   + '_cr_' + str(RF_cr_list[ind_RF]) \
                   + '_lc_' + str(RF_lc_list[ind_RF]) \
                   + '_rc_' + str(RF_rc_list[ind_RF])
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
            _, _, flux_lawson_ignition_piel = get_lawson_criterion_piel(ni, Ti_keV, settings)
            state['flux_mean'] *= settings['cross_section_main_cell']
            # state['flux_mean'] /= flux_lawson
            state['flux_mean'] /= flux_lawson_ignition_piel

            flux_list[ind_N] = state['flux_mean']
            # n1_list[ind_N] = state['n'][-1]
            # n1_list[ind_N] = state['n'][-2]

            selected_number_of_cells = 30
            if number_of_cells == selected_number_of_cells:
                ni_save = state['n']

        except:
            pass

    if RF_lc_list[ind_RF] > 0:
        selectivity = '{:.1f}'.format(RF_rc_list[ind_RF] / RF_lc_list[ind_RF])
        selectivity_trapped = '{:.2f}'.format(RF_cr_list[ind_RF] / RF_cl_list[ind_RF])
    else:
        selectivity = '1'
    label = ''
    label += '$' + 's=' + selectivity + '$'
    # label += ', $' + 's_t=' + selectivity_trapped + '$'
    label += ', $ \\bar{N}_{cl}=' + str(RF_cl_list[ind_RF]) \
             + ', \\bar{N}_{cr}=' + str(RF_cr_list[ind_RF]) \
             + ', \\bar{N}_{lc}=' + str(RF_lc_list[ind_RF]) \
             + ', \\bar{N}_{rc}=' + str(RF_rc_list[ind_RF]) + '$'
    # label = 'set ' + set_name_list[ind_RF]1
    # label = 'set ' + str(int(np.ceil((ind_RF + 1) / 2))) + ' ' + settings['gas_name']
    label = 'set ' + set_name_list[ind_RF]
    print(label)

    # plot flux as a function of N
    plt.figure(1)
    # plt.figure(1, figsize=(7, 7))
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

plt.figure(1)
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
text = '(b)'
plt.text(0.99, 0.98, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

# ax = plt.gca()
# ax.set_xticks([10, 100])
# ax.set_yticks([10, 100])
# # ax.set_yticks([1000, 2000, 4000])
# from matplotlib.ticker import StrMethodFormatter, NullFormatter
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
# plt.legend(loc='lower left')
# plt.legend()
text = '(a)'
# text = '(b)'
plt.text(0.99, 0.98, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

# save pics in high res
# save_dir = '../../../Papers/texts/paper2020/pics/'
save_dir = '../../../Papers/texts/paper2022/pics/'
# save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics/'
# save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics_with_Rm_10/'

# file_name = 'flux_function_of_N'
# # file_name += '_for_poster'
# if RF_type == 'magnetic_transverse':
#     file_name = 'BRF_' + file_name
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')

# file_name = 'n_function_of_cell_number'
# if RF_type == 'magnetic_transverse':
#     file_name = 'BRF_' + file_name
# beingsaved = plt.figure(2)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
