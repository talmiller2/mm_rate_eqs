import matplotlib
# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

import matplotlib.pyplot as plt
from matplotlib import cm

# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 12})

import numpy as np
from scipy.optimize import curve_fit

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from mm_rate_eqs.fusion_functions import get_lawson_parameters, get_lawson_criterion_piel

from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss, get_magnetic_pressure, \
    get_ideal_gas_pressure, get_ideal_gas_energy_per_volume, get_magnetic_field_for_given_pressure, \
    get_bohm_diffusion_constant, get_larmor_radius

plt.close('all')

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs/'

# main_dir += '/runs/slurm_runs/set43_MM_Rm_3_ni_1e21_Ti_10keV_withRMF'
# main_dir += '/runs/slurm_runs/set44_MM_Rm_3_ni_1e21_Ti_10keV_withRMF'
main_dir += '/runs/slurm_runs/set45_MM_Rm_6_ni_1e21_Ti_10keV_withRMF'

# RF_type = 'electric_transverse'
RF_type = 'magnetic_transverse'

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
# colors = ['b', 'b', 'k', 'k', 'g', 'g', 'r', 'r', 'm', 'm', 'k', 'k']
# linestyles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--']

# num_sets = 9
# num_sets = 4

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90]

# linewidth = 1
linewidth = 2

###########################
set_name_list = []
gas_type_list = []
RF_rate_list = []

gas_type = 'deuterium'
# gas_type = 'tritium'


# based on single_particle calcs: set51_B0_1T_l_1m_Post_Rm_6_intervals_D_T

### RMF (BRF=0.04T)

if gas_type == 'deuterium':

    # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["1 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.926, 0.932, 0.015, 0.016, 0.027, 0.038, ]]
    # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["1 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.851, 0.849, 0.072, 0.073, 0.097, 0.067, ]]
    # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["2 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.799, 0.734, 0.018, 0.020, 0.102, 0.121, ]]
    # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["2 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.835, 0.800, 0.071, 0.058, 0.067, 0.097, ]]
    # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["3 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.842, 0.367, 0.021, 0.014, 0.051, 0.021, ]]
    # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["3 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.866, 0.385, 0.060, 0.028, 0.024, 0.046, ]]
    # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["4 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.923, 0.237, 0.018, 0.012, 0.010, 0.004, ]]
    # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["4 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.904, 0.213, 0.057, 0.017, 0.012, 0.025, ]]
    # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["5 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.858, 0.119, 0.020, 0.007, 0.018, 0.015, ]]
    # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["5 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.849, 0.118, 0.047, 0.010, 0.017, 0.016, ]]
    # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["6 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.234, 0.186, 0.009, 0.009, 0.033, 0.039, ]]
    # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["6 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.258, 0.210, 0.011, 0.012, 0.030, 0.029, ]]
    # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["7 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.221, 0.908, 0.009, 0.019, 0.011, 0.005, ]]
    # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["7 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.257, 0.867, 0.015, 0.063, 0.022, 0.039, ]]
    # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["8 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.313, 0.093, 0.011, 0.007, 0.025, 0.011, ]]
    # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["8 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.310, 0.088, 0.015, 0.006, 0.024, 0.009, ]]
    # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["9 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.490, 0.084, 0.013, 0.007, 0.009, 0.011, ]]
    # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["9 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.449, 0.083, 0.021, 0.006, 0.015, 0.008, ]]
    # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["10 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.386, 0.067, 0.012, 0.006, 0.024, 0.020, ]]
    # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["10 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.375, 0.068, 0.017, 0.005, 0.023, 0.013, ]]
    # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["11 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.757, 0.096, 0.021, 0.006, 0.014, 0.004, ]]
    # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["11 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.655, 0.088, 0.035, 0.006, 0.024, 0.010, ]]
    # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
    set_name_list += ["12 (D, iff=1)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.707, 0.696, 0.018, 0.017, 0.138, 0.107, ]]

    # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 0
    set_name_list += ["12 (D, iff=0)"]
    gas_type_list += ["deuterium"]
    RF_rate_list += [[0.742, 0.711, 0.048, 0.039, 0.103, 0.106, ]]


else:
    # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["1 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.489, 0.411, 0.014, 0.011, 0.102, 0.123, ]]
    # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["1 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.512, 0.446, 0.024, 0.025, 0.087, 0.139, ]]
    # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["2 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.505, 0.333, 0.012, 0.013, 0.099, 0.054, ]]
    # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["2 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.558, 0.336, 0.024, 0.017, 0.101, 0.059, ]]
    # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["3 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.629, 0.824, 0.014, 0.015, 0.102, 0.055, ]]
    # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["3 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.688, 0.792, 0.030, 0.049, 0.090, 0.084, ]]
    # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["4 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.615, 0.820, 0.012, 0.014, 0.093, 0.095, ]]
    # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["4 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.633, 0.772, 0.037, 0.053, 0.096, 0.093, ]]
    # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["5 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.659, 0.724, 0.021, 0.023, 0.088, 0.102, ]]
    # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["5 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.694, 0.755, 0.057, 0.061, 0.087, 0.101, ]]
    # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["6 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.937, 0.894, 0.018, 0.018, 0.032, 0.049, ]]
    # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["6 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.839, 0.872, 0.072, 0.066, 0.082, 0.061, ]]
    # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["7 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.814, 0.488, 0.014, 0.011, 0.085, 0.078, ]]
    # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["7 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.814, 0.441, 0.062, 0.030, 0.061, 0.092, ]]
    # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["8 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.841, 0.399, 0.026, 0.015, 0.068, 0.027, ]]
    # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["8 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.877, 0.402, 0.062, 0.032, 0.044, 0.064, ]]
    # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["9 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.892, 0.240, 0.021, 0.012, 0.010, 0.008, ]]
    # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["9 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.894, 0.235, 0.065, 0.020, 0.012, 0.016, ]]
    # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["10 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.860, 0.123, 0.026, 0.008, 0.029, 0.005, ]]
    # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["10 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.809, 0.111, 0.047, 0.010, 0.010, 0.004, ]]
    # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["11 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.622, 0.392, 0.020, 0.017, 0.075, 0.082, ]]
    # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["11 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.595, 0.505, 0.036, 0.034, 0.058, 0.058, ]]
    # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
    set_name_list += ["12 (T, iff=1)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.667, 0.545, 0.016, 0.016, 0.079, 0.107, ]]
    # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 0
    set_name_list += ["12 (T, iff=0)"]
    gas_type_list += ["tritium"]
    RF_rate_list += [[0.652, 0.594, 0.035, 0.035, 0.123, 0.122, ]]

colors = []
linestyles = []
num_sets = int(len(RF_rate_list) / 2)
for i in range(num_sets):
    colors += [cm.rainbow(1.0 * i / num_sets)]
    colors += [cm.rainbow(1.0 * i / num_sets)]
    linestyles += ['-', '--']

for ind_RF in range(len(RF_rate_list)):
    # for ind_RF in range(4):
    print('ind_RF:', ind_RF)

    RF_rc_curr = RF_rate_list[ind_RF][0]
    RF_lc_curr = RF_rate_list[ind_RF][1]
    RF_cr_curr = RF_rate_list[ind_RF][2]
    RF_cl_curr = RF_rate_list[ind_RF][3]
    RF_rl_curr = RF_rate_list[ind_RF][4]
    RF_lr_curr = RF_rate_list[ind_RF][5]

    color = colors[ind_RF]
    linestyle = linestyles[ind_RF]
    plasma_mode = 'isoT'

    flux_list = np.nan * np.zeros(len(num_cells_list))
    n1_list = np.nan * np.zeros(len(num_cells_list))
    for ind_N, number_of_cells in enumerate(num_cells_list):
        run_name = plasma_mode
        run_name += '_' + gas_type_list[ind_RF]
        RF_label = 'RF_terms' \
                   + '_rc_' + str(RF_rc_curr) \
                   + '_lc_' + str(RF_lc_curr) \
                   + '_cr_' + str(RF_cr_curr) \
                   + '_cl_' + str(RF_cl_curr) \
                   + '_rl_' + str(RF_rl_curr) \
                   + '_lr_' + str(RF_lr_curr)
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

    # if RF_lc_list[ind_RF] > 0:
    #     selectivity = '{:.1f}'.format(RF_rc_list[ind_RF] / RF_lc_list[ind_RF])
    #     selectivity_trapped = '{:.2f}'.format(RF_cr_list[ind_RF] / RF_cl_list[ind_RF])
    # else:
    #     selectivity = '1'
    # label = ''
    # label += '$' + 's=' + selectivity + '$'
    # # label += ', $' + 's_t=' + selectivity_trapped + '$'
    # label += ', $ \\bar{N}_{cl}=' + str(RF_cl_list[ind_RF]) \
    #          + ', \\bar{N}_{cr}=' + str(RF_cr_list[ind_RF]) \
    #          + ', \\bar{N}_{lc}=' + str(RF_lc_list[ind_RF]) \
    #          + ', \\bar{N}_{rc}=' + str(RF_rc_list[ind_RF]) + '$'
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
plt.title('RF ' + RF_type + ', gas ' + gas_type)
plt.tight_layout()
plt.legend(ncols=2)
# text = '(a)'
text = '(b)'
# plt.text(0.99, 0.98, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
#          horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

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
# plt.text(0.99, 0.98, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 30},
#          horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

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
