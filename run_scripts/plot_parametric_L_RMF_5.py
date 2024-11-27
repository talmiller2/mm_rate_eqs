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
# main_dir += '/runs/slurm_runs/set45_MM_Rm_6_ni_1e21_Ti_10keV_withRMF'
main_dir += '/runs/slurm_runs/set46_MM_Rm_10_ni_1e21_Ti_10keV_withRMF'

RF_type = 'electric_transverse'
# RF_type = 'magnetic_transverse'

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90]

# linewidth = 1
linewidth = 2

###########################
set_name_list = []
gas_type_list = []
RF_rate_list = []

# gas_type = 'deuterium'
gas_type = 'tritium'

RF_type = 'electric_transverse'
# RF_type = 'magnetic_transverse'

if RF_type == 'magnetic_transverse':

    # based on single_particle calcs: set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T

    ### RMF (BRF=0.04T)

    if gas_type == 'deuterium':

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["1 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.933, 0.964, 0.012, 0.010, 0.037, 0.019, ]]
        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["1 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.920, 0.879, 0.044, 0.045, 0.038, 0.055, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["2 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.912, 0.447, 0.016, 0.009, 0.037, 0.024, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["2 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.907, 0.416, 0.040, 0.016, 0.027, 0.033, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["3 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.915, 0.257, 0.012, 0.005, 0.005, 0.003, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["3 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.921, 0.236, 0.033, 0.010, 0.008, 0.020, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["4 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.943, 0.171, 0.015, 0.004, 0.000, 0.005, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["4 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.866, 0.185, 0.033, 0.007, 0.024, 0.015, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["5 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.947, 0.927, 0.009, 0.012, 0.022, 0.038, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["5 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.911, 0.911, 0.048, 0.045, 0.032, 0.049, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["6 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.869, 0.703, 0.012, 0.014, 0.037, 0.037, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["6 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.887, 0.714, 0.043, 0.026, 0.042, 0.042, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["7 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.305, 0.338, 0.007, 0.007, 0.047, 0.045, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["7 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.296, 0.345, 0.008, 0.009, 0.057, 0.050, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["8 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.435, 0.160, 0.010, 0.005, 0.027, 0.021, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["8 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.414, 0.153, 0.011, 0.004, 0.013, 0.010, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["9 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.489, 0.142, 0.010, 0.003, 0.016, 0.023, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["9 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.517, 0.111, 0.012, 0.004, 0.010, 0.026, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["10 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.506, 0.117, 0.011, 0.005, 0.024, 0.038, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["10 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.457, 0.121, 0.012, 0.003, 0.011, 0.004, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["11 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.275, 0.753, 0.005, 0.014, 0.043, 0.031, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["11 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.255, 0.763, 0.009, 0.020, 0.046, 0.023, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["12 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.263, 0.931, 0.005, 0.017, 0.004, 0.007, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["12 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.264, 0.896, 0.009, 0.034, 0.031, 0.010, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["13 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.693, 0.699, 0.012, 0.015, 0.090, 0.062, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["13 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.626, 0.679, 0.020, 0.020, 0.113, 0.097, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["14 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.746, 0.201, 0.013, 0.005, 0.026, 0.022, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["14 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.710, 0.176, 0.019, 0.008, 0.028, 0.042, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["15 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.741, 0.140, 0.014, 0.004, 0.011, 0.032, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["15 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.745, 0.166, 0.020, 0.005, 0.007, 0.019, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["16 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.739, 0.158, 0.012, 0.004, 0.023, 0.021, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["16 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.725, 0.105, 0.018, 0.004, 0.017, 0.042, ]]

    else:

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["1 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.628, 0.567, 0.012, 0.012, 0.113, 0.093, ]]
        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["1 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.677, 0.631, 0.018, 0.020, 0.084, 0.079, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["2 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.835, 0.888, 0.008, 0.012, 0.054, 0.047, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["2 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.754, 0.825, 0.028, 0.034, 0.087, 0.099, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["3 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.761, 0.858, 0.010, 0.012, 0.055, 0.083, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["3 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.791, 0.881, 0.030, 0.039, 0.072, 0.042, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["4 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.702, 0.777, 0.016, 0.015, 0.146, 0.096, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["4 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.819, 0.755, 0.033, 0.036, 0.067, 0.120, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["5 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.545, 0.461, 0.010, 0.012, 0.139, 0.121, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["5 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.642, 0.417, 0.019, 0.015, 0.093, 0.069, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["6 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.582, 0.433, 0.010, 0.007, 0.115, 0.055, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["6 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.651, 0.403, 0.022, 0.018, 0.122, 0.101, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["7 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.908, 0.950, 0.011, 0.014, 0.046, 0.036, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["7 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.907, 0.922, 0.048, 0.045, 0.053, 0.051, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["8 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.887, 0.442, 0.015, 0.009, 0.040, 0.034, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["8 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.889, 0.470, 0.039, 0.019, 0.054, 0.044, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["9 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.897, 0.247, 0.016, 0.006, 0.034, 0.006, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["9 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.920, 0.267, 0.035, 0.009, 0.021, 0.005, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["10 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.890, 0.198, 0.014, 0.006, 0.021, 0.009, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["10 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.851, 0.177, 0.028, 0.007, 0.028, 0.005, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["11 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.880, 0.856, 0.015, 0.013, 0.087, 0.059, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["11 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.846, 0.869, 0.042, 0.038, 0.103, 0.045, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["12 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.855, 0.727, 0.012, 0.011, 0.090, 0.072, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["12 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.872, 0.817, 0.038, 0.037, 0.060, 0.051, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["13 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.853, 0.893, 0.012, 0.012, 0.069, 0.057, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["13 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.818, 0.790, 0.029, 0.031, 0.074, 0.090, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["14 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.875, 0.826, 0.015, 0.012, 0.060, 0.083, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["14 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.860, 0.877, 0.049, 0.042, 0.052, 0.071, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["15 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.845, 0.724, 0.016, 0.014, 0.082, 0.063, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["15 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.829, 0.793, 0.040, 0.032, 0.109, 0.060, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["16 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.860, 0.301, 0.019, 0.010, 0.018, 0.026, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["16 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.915, 0.297, 0.040, 0.012, 0.005, 0.026, ]]

    colors = []
    linestyles = []
    num_sets = int(len(RF_rate_list) / 2)
    for i in range(num_sets):
        colors += [cm.rainbow(1.0 * i / num_sets)]
        colors += [cm.rainbow(1.0 * i / num_sets)]
        linestyles += ['-', '--']

if RF_type == 'electric_transverse':
    if gas_type == 'deuterium':
        ### REF (ERF=50kV/m)

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["1 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.957, 0.933, 0.008, 0.008, 0.014, 0.047, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["2 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.883, 0.518, 0.014, 0.014, 0.053, 0.024, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["3 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.951, 0.313, 0.016, 0.009, 0.019, 0.013, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["4 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.958, 0.211, 0.016, 0.007, 0.003, 0.009, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["5 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.891, 0.902, 0.008, 0.008, 0.039, 0.044, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["6 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.871, 0.690, 0.009, 0.009, 0.032, 0.033, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["7 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.569, 0.362, 0.012, 0.010, 0.031, 0.037, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["8 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.694, 0.138, 0.015, 0.005, 0.026, 0.013, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["9 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.690, 0.096, 0.014, 0.004, 0.031, 0.013, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["10 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.682, 0.091, 0.015, 0.004, 0.033, 0.005, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["11 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.511, 0.866, 0.012, 0.020, 0.037, 0.017, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["12 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.434, 0.937, 0.011, 0.016, 0.025, 0.006, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["13 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.820, 0.774, 0.015, 0.015, 0.083, 0.144, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["14 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.855, 0.238, 0.017, 0.009, 0.029, 0.042, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["15 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.854, 0.191, 0.017, 0.006, 0.029, 0.012, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["16 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.870, 0.142, 0.017, 0.005, 0.018, 0.018, ]]

    else:
        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["1 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.720, 0.722, 0.014, 0.012, 0.127, 0.059, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["2 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.755, 0.888, 0.009, 0.007, 0.040, 0.057, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["3 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.790, 0.909, 0.008, 0.010, 0.082, 0.059, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["4 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.780, 0.787, 0.012, 0.013, 0.049, 0.086, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["5 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.776, 0.609, 0.012, 0.010, 0.063, 0.048, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["6 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.729, 0.643, 0.013, 0.010, 0.102, 0.081, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["7 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.942, 0.953, 0.006, 0.006, 0.040, 0.018, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["8 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.917, 0.730, 0.013, 0.011, 0.034, 0.014, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["9 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.978, 0.373, 0.014, 0.009, 0.002, 0.011, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["10 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.958, 0.277, 0.016, 0.007, 0.018, 0.005, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["11 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.934, 0.923, 0.007, 0.008, 0.037, 0.033, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["12 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.859, 0.831, 0.008, 0.010, 0.065, 0.054, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["13 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.931, 0.901, 0.010, 0.008, 0.031, 0.043, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["14 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.918, 0.928, 0.008, 0.011, 0.043, 0.034, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["15 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.855, 0.759, 0.011, 0.009, 0.049, 0.064, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["16 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.873, 0.474, 0.011, 0.010, 0.008, 0.029, ]]

    colors = []
    linestyles = []
    num_sets = len(RF_rate_list)
    for i in range(num_sets):
        colors += [cm.rainbow(1.0 * i / num_sets)]
        linestyles += ['-']

####

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
plt.legend(ncols=3)
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
