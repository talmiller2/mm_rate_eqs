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
main_dir += '/runs/slurm_runs/set44_MM_Rm_3_ni_1e21_Ti_10keV_withRMF'

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

RF_type = 'electric_transverse'
# RF_type = 'magnetic_transverse'

# gas_type = 'deuterium'
gas_type = 'tritium'

# based on single_particle calcs: set50_B0_1T_l_1m_Post_Rm_3_intervals_D_T
### RMF

if RF_type == 'magnetic_transverse':
    if gas_type == 'deuterium':
        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["1 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.862, 0.839, 0.028, 0.027, 0.061, 0.072, ]]
        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["1 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.731, 0.742, 0.127, 0.128, 0.136, 0.117, ]]
        # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["2 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.724, 0.539, 0.025, 0.021, 0.121, 0.101, ]]
        # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["2 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.735, 0.598, 0.121, 0.080, 0.101, 0.115, ]]
        # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["3 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.838, 0.179, 0.033, 0.016, 0.014, 0.020, ]]
        # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["3 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.750, 0.220, 0.112, 0.033, 0.030, 0.043, ]]
        # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["4 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.797, 0.130, 0.036, 0.011, 0.028, 0.024, ]]
        # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["4 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.734, 0.168, 0.099, 0.023, 0.019, 0.036, ]]
        # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["5 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.827, 0.102, 0.035, 0.009, 0.006, 0.017, ]]
        # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["5 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.704, 0.108, 0.094, 0.016, 0.018, 0.028, ]]
        # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["6 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.130, 0.149, 0.014, 0.013, 0.012, 0.024, ]]
        # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["6 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.166, 0.177, 0.019, 0.021, 0.016, 0.017, ]]
        # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["7 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.118, 0.829, 0.013, 0.033, 0.017, 0.017, ]]
        # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["7 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.122, 0.736, 0.022, 0.102, 0.046, 0.022, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["8 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.220, 0.101, 0.019, 0.011, 0.021, 0.017, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["8 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.214, 0.090, 0.023, 0.011, 0.017, 0.022, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["9 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.257, 0.074, 0.019, 0.009, 0.017, 0.023, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["9 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.254, 0.083, 0.025, 0.008, 0.010, 0.025, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["10 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.260, 0.085, 0.020, 0.009, 0.021, 0.018, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["10 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.253, 0.078, 0.026, 0.007, 0.020, 0.023, ]]
        # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["11 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.546, 0.084, 0.026, 0.008, 0.013, 0.021, ]]
        # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["11 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.481, 0.087, 0.052, 0.010, 0.011, 0.027, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["12 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.418, 0.408, 0.025, 0.021, 0.069, 0.065, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["12 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.390, 0.419, 0.050, 0.047, 0.068, 0.094, ]]

    else:

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["1 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.294, 0.244, 0.011, 0.017, 0.059, 0.071, ]]
        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["1 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.380, 0.350, 0.037, 0.048, 0.108, 0.098, ]]
        # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["2 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.272, 0.166, 0.011, 0.008, 0.053, 0.037, ]]
        # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["2 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.322, 0.253, 0.044, 0.034, 0.089, 0.070, ]]
        # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["3 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.350, 0.667, 0.015, 0.024, 0.054, 0.097, ]]
        # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["3 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.421, 0.622, 0.047, 0.081, 0.085, 0.106, ]]
        # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["4 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.361, 0.725, 0.019, 0.026, 0.060, 0.078, ]]
        # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["4 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.401, 0.692, 0.043, 0.109, 0.100, 0.094, ]]
        # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["5 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.295, 0.517, 0.017, 0.025, 0.045, 0.082, ]]
        # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["5 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.378, 0.638, 0.052, 0.112, 0.050, 0.069, ]]
        # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["6 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.831, 0.825, 0.028, 0.029, 0.083, 0.074, ]]
        # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["6 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.770, 0.730, 0.122, 0.133, 0.113, 0.114, ]]
        # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["7 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.668, 0.308, 0.024, 0.015, 0.103, 0.041, ]]
        # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["7 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.674, 0.391, 0.116, 0.055, 0.095, 0.077, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["8 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.826, 0.206, 0.038, 0.014, 0.024, 0.017, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["8 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.802, 0.236, 0.125, 0.037, 0.021, 0.032, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["9 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.791, 0.130, 0.037, 0.012, 0.026, 0.019, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["9 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.766, 0.157, 0.102, 0.022, 0.022, 0.034, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["10 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.771, 0.097, 0.039, 0.008, 0.012, 0.014, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["10 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.695, 0.104, 0.087, 0.016, 0.016, 0.030, ]]
        # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["11 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.487, 0.147, 0.035, 0.015, 0.028, 0.024, ]]
        # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["11 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.537, 0.211, 0.077, 0.029, 0.032, 0.034, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["12 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.505, 0.472, 0.023, 0.022, 0.084, 0.078, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["12 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.551, 0.479, 0.073, 0.069, 0.119, 0.112, ]]

    colors = []
    linestyles = []
    num_sets = int(len(RF_rate_list) / 2)
    for i in range(num_sets):
        colors += [cm.rainbow(1.0 * i / num_sets)]
        colors += [cm.rainbow(1.0 * i / num_sets)]
        linestyles += ['-', '--']

if RF_type == 'electric_transverse':
    if gas_type == 'deuterium':

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["1 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.886, 0.901, 0.006, 0.009, 0.047, 0.046, ]]
        # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["2 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.756, 0.620, 0.014, 0.016, 0.096, 0.059, ]]
        # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["3 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.911, 0.314, 0.030, 0.019, 0.007, 0.016, ]]
        # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["4 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.898, 0.193, 0.035, 0.016, 0.010, 0.009, ]]
        # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["5 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.884, 0.125, 0.042, 0.012, 0.012, 0.007, ]]
        # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["6 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.205, 0.179, 0.019, 0.017, 0.030, 0.025, ]]
        # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["7 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.203, 0.884, 0.017, 0.038, 0.011, 0.016, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["8 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.294, 0.087, 0.024, 0.011, 0.016, 0.017, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["9 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.316, 0.074, 0.023, 0.010, 0.015, 0.012, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["10 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.319, 0.092, 0.027, 0.009, 0.036, 0.016, ]]
        # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["11 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.711, 0.088, 0.043, 0.010, 0.011, 0.010, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["12 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.533, 0.496, 0.033, 0.033, 0.088, 0.118, ]]

    else:

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["1 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.412, 0.470, 0.017, 0.018, 0.068, 0.092, ]]
        # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["2 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.415, 0.338, 0.015, 0.014, 0.095, 0.068, ]]
        # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["3 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.515, 0.823, 0.016, 0.018, 0.061, 0.082, ]]
        # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["4 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.503, 0.783, 0.012, 0.012, 0.046, 0.072, ]]
        # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["5 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.486, 0.650, 0.013, 0.018, 0.061, 0.069, ]]
        # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["6 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.893, 0.913, 0.008, 0.007, 0.042, 0.044, ]]
        # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["7 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.782, 0.467, 0.014, 0.013, 0.064, 0.052, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["8 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.903, 0.357, 0.023, 0.020, 0.014, 0.023, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["9 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.882, 0.223, 0.033, 0.015, 0.017, 0.015, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["10 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.864, 0.149, 0.046, 0.012, 0.014, 0.010, ]]
        # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["11 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.607, 0.294, 0.025, 0.016, 0.027, 0.026, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["12 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.693, 0.648, 0.023, 0.019, 0.084, 0.072, ]]

    colors = []
    linestyles = []
    num_sets = len(RF_rate_list)
    for i in range(num_sets):
        colors += [cm.rainbow(1.0 * i / num_sets)]
        linestyles += ['-']

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
