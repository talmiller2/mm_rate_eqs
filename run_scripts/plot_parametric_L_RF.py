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

# num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90]
num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

# linewidth = 1
linewidth = 2

###########################
set_name_list = []
gas_type_list = []
RF_rate_list = []

# gas_type = 'deuterium'
gas_type = 'tritium'

# RF_type = 'electric_transverse'
RF_type = 'magnetic_transverse'

if gas_type == 'deuterium':
    gas_type_short = 'D'
else:
    gas_type_short = 'T'

if RF_type == 'magnetic_transverse':
    RF_type_short = 'RMF'
else:
    RF_type_short = 'REF'

if RF_type == 'magnetic_transverse':

    # based on single_particle calcs: set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T

    if gas_type == 'deuterium':
        ### RMF (BRF=0.02T)

        # # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        # set_name_list += ["1 (D, iff=1)"]
        # gas_type_list += ["deuterium"]
        # RF_rate_list += [[0.804, 0.765, 0.018, 0.014, 0.087, 0.127, ]]
        # # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        # set_name_list += ["1 (D, iff=0)"]
        # gas_type_list += ["deuterium"]
        # RF_rate_list += [[0.865, 0.847, 0.035, 0.038, 0.053, 0.089, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["2 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.771, 0.226, 0.015, 0.006, 0.030, 0.024, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["2 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.787, 0.237, 0.027, 0.007, 0.024, 0.035, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["3 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.789, 0.107, 0.013, 0.004, 0.018, 0.011, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["3 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.793, 0.121, 0.022, 0.004, 0.020, 0.006, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["4 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.838, 0.087, 0.019, 0.002, 0.005, 0.000, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["4 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.808, 0.109, 0.022, 0.003, 0.016, 0.005, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["5 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.833, 0.784, 0.014, 0.014, 0.065, 0.072, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["5 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.789, 0.786, 0.032, 0.027, 0.104, 0.092, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["6 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.751, 0.306, 0.015, 0.010, 0.069, 0.056, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["6 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.749, 0.405, 0.026, 0.019, 0.092, 0.069, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["7 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.173, 0.160, 0.004, 0.004, 0.030, 0.030, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["7 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.190, 0.153, 0.006, 0.004, 0.022, 0.026, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["8 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.314, 0.059, 0.006, 0.002, 0.004, 0.001, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["8 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.278, 0.079, 0.006, 0.003, 0.013, 0.023, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["9 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.260, 0.054, 0.006, 0.002, 0.011, 0.008, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["9 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.331, 0.061, 0.010, 0.002, 0.013, 0.013, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["10 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.317, 0.068, 0.006, 0.003, 0.020, 0.012, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["10 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.319, 0.043, 0.008, 0.002, 0.025, 0.025, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["11 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.091, 0.624, 0.004, 0.014, 0.061, 0.021, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["11 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.136, 0.597, 0.004, 0.016, 0.025, 0.025, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["12 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.134, 0.774, 0.004, 0.015, 0.053, 0.007, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["12 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.145, 0.791, 0.004, 0.024, 0.019, 0.019, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["13 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.372, 0.400, 0.009, 0.008, 0.046, 0.090, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["13 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.444, 0.436, 0.013, 0.013, 0.079, 0.119, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["14 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.507, 0.067, 0.012, 0.002, 0.014, 0.026, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["14 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.561, 0.077, 0.016, 0.004, 0.017, 0.029, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["15 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.495, 0.095, 0.012, 0.002, 0.011, 0.008, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["15 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.582, 0.075, 0.019, 0.003, 0.019, 0.012, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["16 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.609, 0.060, 0.012, 0.001, 0.017, 0.003, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
        set_name_list += ["16 (D, iff=0)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.606, 0.054, 0.014, 0.002, 0.014, 0.010, ]]

    else:

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["1 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.315, 0.373, 0.007, 0.008, 0.095, 0.066, ]]
        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["1 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.422, 0.372, 0.011, 0.009, 0.099, 0.074, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["2 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.403, 0.705, 0.009, 0.016, 0.037, 0.124, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["2 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.441, 0.774, 0.016, 0.026, 0.098, 0.101, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["3 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.491, 0.775, 0.011, 0.012, 0.126, 0.066, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["3 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.476, 0.718, 0.017, 0.026, 0.044, 0.122, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["4 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.494, 0.549, 0.013, 0.015, 0.086, 0.057, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["4 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.503, 0.601, 0.017, 0.019, 0.109, 0.061, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["5 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.397, 0.204, 0.008, 0.005, 0.077, 0.058, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["5 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.467, 0.253, 0.014, 0.007, 0.082, 0.054, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["6 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.463, 0.215, 0.009, 0.006, 0.064, 0.037, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["6 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.434, 0.179, 0.013, 0.005, 0.049, 0.031, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["7 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.723, 0.814, 0.016, 0.017, 0.169, 0.116, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["7 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.889, 0.763, 0.035, 0.038, 0.077, 0.110, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["8 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.779, 0.194, 0.016, 0.004, 0.040, 0.041, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["8 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.765, 0.216, 0.024, 0.006, 0.045, 0.070, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["9 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.777, 0.096, 0.016, 0.003, 0.020, 0.013, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["9 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.761, 0.126, 0.023, 0.003, 0.017, 0.026, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["10 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.724, 0.082, 0.017, 0.002, 0.030, 0.006, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["10 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.762, 0.072, 0.022, 0.003, 0.007, 0.007, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["11 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.764, 0.615, 0.016, 0.013, 0.120, 0.068, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["11 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.725, 0.525, 0.029, 0.018, 0.119, 0.090, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["12 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.635, 0.399, 0.014, 0.007, 0.092, 0.064, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["12 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.658, 0.407, 0.027, 0.015, 0.158, 0.086, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["13 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.755, 0.617, 0.011, 0.015, 0.072, 0.147, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["13 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.745, 0.639, 0.022, 0.019, 0.121, 0.102, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["14 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.759, 0.613, 0.015, 0.015, 0.092, 0.187, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["14 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.697, 0.671, 0.025, 0.025, 0.131, 0.115, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["15 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.649, 0.346, 0.019, 0.010, 0.133, 0.084, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["15 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.708, 0.402, 0.023, 0.016, 0.098, 0.069, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["16 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.703, 0.140, 0.016, 0.005, 0.018, 0.037, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
        set_name_list += ["16 (T, iff=0)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.728, 0.152, 0.023, 0.006, 0.016, 0.026, ]]

    colors = []
    linestyles = []
    num_sets = int(len(RF_rate_list) / 2)
    for i in range(num_sets):
        colors += [cm.rainbow(1.0 * i / num_sets)]
        colors += [cm.rainbow(1.0 * i / num_sets)]
        linestyles += ['-', '--']

if RF_type == 'electric_transverse':
    if gas_type == 'deuterium':
        ### REF (ERF=25kV/m)

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["1 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.857, 0.847, 0.013, 0.013, 0.050, 0.093, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["2 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.851, 0.273, 0.018, 0.009, 0.026, 0.006, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["3 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.821, 0.128, 0.022, 0.006, 0.013, 0.012, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["4 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.866, 0.058, 0.020, 0.005, 0.012, 0.010, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["5 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.788, 0.761, 0.012, 0.016, 0.053, 0.091, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["6 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.726, 0.459, 0.014, 0.011, 0.057, 0.042, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["7 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.325, 0.154, 0.007, 0.004, 0.039, 0.011, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["8 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.388, 0.050, 0.007, 0.003, 0.018, 0.007, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["9 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.384, 0.041, 0.008, 0.002, 0.052, 0.001, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["10 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.410, 0.052, 0.009, 0.003, 0.018, 0.001, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["11 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.293, 0.645, 0.008, 0.015, 0.054, 0.031, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["12 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.267, 0.803, 0.007, 0.016, 0.014, 0.023, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["13 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.570, 0.484, 0.014, 0.011, 0.089, 0.078, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["14 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.713, 0.085, 0.017, 0.004, 0.023, 0.011, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["15 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.693, 0.060, 0.015, 0.003, 0.019, 0.012, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
        set_name_list += ["16 (D, iff=1)"]
        gas_type_list += ["deuterium"]
        RF_rate_list += [[0.614, 0.054, 0.013, 0.003, 0.033, 0.006, ]]

    else:

        # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["1 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.501, 0.455, 0.012, 0.010, 0.086, 0.117, ]]
        # set 2 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["2 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.517, 0.815, 0.015, 0.015, 0.101, 0.092, ]]
        # set 3 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["3 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.575, 0.787, 0.010, 0.011, 0.037, 0.112, ]]
        # set 4 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["4 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.625, 0.603, 0.016, 0.013, 0.074, 0.082, ]]
        # set 5 , alpha= 1.42 , omega/omega0= 1.703 , beta= 0.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["5 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.550, 0.403, 0.010, 0.013, 0.115, 0.057, ]]
        # set 6 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["6 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.571, 0.414, 0.012, 0.009, 0.052, 0.037, ]]
        # set 7 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["7 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.851, 0.867, 0.014, 0.014, 0.083, 0.062, ]]
        # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["8 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.836, 0.366, 0.020, 0.009, 0.038, 0.025, ]]
        # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["9 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.795, 0.166, 0.021, 0.007, 0.005, 0.013, ]]
        # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["10 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.820, 0.081, 0.022, 0.005, 0.013, 0.010, ]]
        # set 11 , alpha= 1.0 , omega/omega0= 1.199 , beta= 0.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["11 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.796, 0.626, 0.014, 0.014, 0.069, 0.116, ]]
        # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["12 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.813, 0.639, 0.014, 0.011, 0.046, 0.052, ]]
        # set 13 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["13 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.747, 0.782, 0.014, 0.018, 0.123, 0.111, ]]
        # set 14 , alpha= 0.94 , omega/omega0= 1.127 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["14 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.779, 0.799, 0.016, 0.013, 0.051, 0.064, ]]
        # set 15 , alpha= 0.88 , omega/omega0= 1.055 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["15 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.691, 0.542, 0.019, 0.016, 0.071, 0.093, ]]
        # set 16 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
        set_name_list += ["16 (T, iff=1)"]
        gas_type_list += ["tritium"]
        RF_rate_list += [[0.749, 0.286, 0.018, 0.008, 0.008, 0.009, ]]

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
            _, flux_lawson_piel, _, flux_lawson_ignition_piel = get_lawson_criterion_piel(ni, Ti_keV, settings)
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
# plt.title('RF ' + RF_type + ', gas ' + gas_type)
plt.title(RF_type_short + ' (' + gas_type_short + ')')
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
