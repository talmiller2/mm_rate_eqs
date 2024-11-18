import os

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

n0 = 1e21  # m^-3
Ti = 10 * 1e3  # eV
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set41_MM_Rm_3_ni_1e21_Ti_10keV_withRF'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set42_MM_Rm_3_ni_1e21_Ti_10keV_withRF'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set43_MM_Rm_3_ni_1e21_Ti_10keV_withRMF'
main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set44_MM_Rm_3_ni_1e21_Ti_10keV_withRMF'

slurm_kwargs = {}
slurm_kwargs['partition'] = 'core'
# slurm_kwargs['partition'] = 'testCore'
# slurm_kwargs['partition'] = 'socket'
# slurm_kwargs['partition'] = 'testSocket'
slurm_kwargs['ntasks'] = 1
slurm_kwargs['cpus-per-task'] = 1

plasma_mode = 'isoT'

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

###########################
set_name_list = []
gas_type_list = []
RF_rate_list = []

######################

# based on single_particle calcs: set50_B0_1T_l_1m_Post_Rm_3_intervals_D_T

### RMF

# # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["1 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.862, 0.839, 0.028, 0.027, 0.061, 0.072, ]]
# # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["1 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.731, 0.742, 0.127, 0.128, 0.136, 0.117, ]]
# # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["2 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.724, 0.539, 0.025, 0.021, 0.121, 0.101, ]]
# # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["2 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.735, 0.598, 0.121, 0.080, 0.101, 0.115, ]]
# # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["3 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.838, 0.179, 0.033, 0.016, 0.014, 0.020, ]]
# # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["3 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.750, 0.220, 0.112, 0.033, 0.030, 0.043, ]]
# # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["4 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.797, 0.130, 0.036, 0.011, 0.028, 0.024, ]]
# # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["4 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.734, 0.168, 0.099, 0.023, 0.019, 0.036, ]]
# # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["5 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.827, 0.102, 0.035, 0.009, 0.006, 0.017, ]]
# # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["5 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.704, 0.108, 0.094, 0.016, 0.018, 0.028, ]]
# # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["6 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.130, 0.149, 0.014, 0.013, 0.012, 0.024, ]]
# # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["6 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.166, 0.177, 0.019, 0.021, 0.016, 0.017, ]]
# # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["7 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.118, 0.829, 0.013, 0.033, 0.017, 0.017, ]]
# # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["7 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.122, 0.736, 0.022, 0.102, 0.046, 0.022, ]]
# # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["8 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.220, 0.101, 0.019, 0.011, 0.021, 0.017, ]]
# # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["8 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.214, 0.090, 0.023, 0.011, 0.017, 0.022, ]]
# # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["9 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.257, 0.074, 0.019, 0.009, 0.017, 0.023, ]]
# # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["9 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.254, 0.083, 0.025, 0.008, 0.010, 0.025, ]]
# # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["10 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.260, 0.085, 0.020, 0.009, 0.021, 0.018, ]]
# # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["10 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.253, 0.078, 0.026, 0.007, 0.020, 0.023, ]]
# # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["11 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.546, 0.084, 0.026, 0.008, 0.013, 0.021, ]]
# # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["11 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.481, 0.087, 0.052, 0.010, 0.011, 0.027, ]]
# # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
# set_name_list += ["12 (D, iff=1)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.418, 0.408, 0.025, 0.021, 0.069, 0.065, ]]
# # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 0
# set_name_list += ["12 (D, iff=0)"]
# gas_type_list += ["deuterium"]
# RF_rate_list += [[0.390, 0.419, 0.050, 0.047, 0.068, 0.094, ]]
#
# # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["1 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.294, 0.244, 0.011, 0.017, 0.059, 0.071, ]]
# # set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["1 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.380, 0.350, 0.037, 0.048, 0.108, 0.098, ]]
# # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["2 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.272, 0.166, 0.011, 0.008, 0.053, 0.037, ]]
# # set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["2 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.322, 0.253, 0.044, 0.034, 0.089, 0.070, ]]
# # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["3 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.350, 0.667, 0.015, 0.024, 0.054, 0.097, ]]
# # set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["3 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.421, 0.622, 0.047, 0.081, 0.085, 0.106, ]]
# # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["4 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.361, 0.725, 0.019, 0.026, 0.060, 0.078, ]]
# # set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["4 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.401, 0.692, 0.043, 0.109, 0.100, 0.094, ]]
# # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["5 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.295, 0.517, 0.017, 0.025, 0.045, 0.082, ]]
# # set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["5 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.378, 0.638, 0.052, 0.112, 0.050, 0.069, ]]
# # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["6 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.831, 0.825, 0.028, 0.029, 0.083, 0.074, ]]
# # set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["6 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.770, 0.730, 0.122, 0.133, 0.113, 0.114, ]]
# # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["7 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.668, 0.308, 0.024, 0.015, 0.103, 0.041, ]]
# # set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["7 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.674, 0.391, 0.116, 0.055, 0.095, 0.077, ]]
# # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["8 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.826, 0.206, 0.038, 0.014, 0.024, 0.017, ]]
# # set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["8 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.802, 0.236, 0.125, 0.037, 0.021, 0.032, ]]
# # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["9 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.791, 0.130, 0.037, 0.012, 0.026, 0.019, ]]
# # set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["9 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.766, 0.157, 0.102, 0.022, 0.022, 0.034, ]]
# # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["10 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.771, 0.097, 0.039, 0.008, 0.012, 0.014, ]]
# # set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["10 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.695, 0.104, 0.087, 0.016, 0.016, 0.030, ]]
# # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["11 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.487, 0.147, 0.035, 0.015, 0.028, 0.024, ]]
# # set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["11 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.537, 0.211, 0.077, 0.029, 0.032, 0.034, ]]
# # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
# set_name_list += ["12 (T, iff=1)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.505, 0.472, 0.023, 0.022, 0.084, 0.078, ]]
# # set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 0
# set_name_list += ["12 (T, iff=0)"]
# gas_type_list += ["tritium"]
# RF_rate_list += [[0.551, 0.479, 0.073, 0.069, 0.119, 0.112, ]]


### REF (iff=0,1 are almost identical)

# set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["1 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.886, 0.901, 0.006, 0.009, 0.047, 0.046, ]]
#set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["2 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.756, 0.620, 0.014, 0.016, 0.096, 0.059, ]]
#set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["3 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.911, 0.314, 0.030, 0.019, 0.007, 0.016, ]]
#set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["4 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.898, 0.193, 0.035, 0.016, 0.010, 0.009, ]]
#set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["5 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.884, 0.125, 0.042, 0.012, 0.012, 0.007, ]]
#set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["6 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.205, 0.179, 0.019, 0.017, 0.030, 0.025, ]]
#set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["7 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.203, 0.884, 0.017, 0.038, 0.011, 0.016, ]]
#set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["8 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.294, 0.087, 0.024, 0.011, 0.016, 0.017, ]]
#set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["9 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.316, 0.074, 0.023, 0.010, 0.015, 0.012, ]]
#set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["10 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.319, 0.092, 0.027, 0.009, 0.036, 0.016, ]]
#set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["11 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.711, 0.088, 0.043, 0.010, 0.011, 0.010, ]]
#set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["12 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.533, 0.496, 0.033, 0.033, 0.088, 0.118, ]]

#set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["1 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.412, 0.470, 0.017, 0.018, 0.068, 0.092, ]]
#set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
set_name_list += ["2 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.415, 0.338, 0.015, 0.014, 0.095, 0.068, ]]
#set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["3 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.515, 0.823, 0.016, 0.018, 0.061, 0.082, ]]
#set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
set_name_list += ["4 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.503, 0.783, 0.012, 0.012, 0.046, 0.072, ]]
#set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
set_name_list += ["5 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.486, 0.650, 0.013, 0.018, 0.061, 0.069, ]]
#set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["6 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.893, 0.913, 0.008, 0.007, 0.042, 0.044, ]]
#set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
set_name_list += ["7 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.782, 0.467, 0.014, 0.013, 0.064, 0.052, ]]
#set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["8 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.903, 0.357, 0.023, 0.020, 0.014, 0.023, ]]
#set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
set_name_list += ["9 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.882, 0.223, 0.033, 0.015, 0.017, 0.015, ]]
#set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
set_name_list += ["10 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.864, 0.149, 0.046, 0.012, 0.014, 0.010, ]]
#set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
set_name_list += ["11 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.607, 0.294, 0.025, 0.016, 0.027, 0.026, ]]
#set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["12 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.693, 0.648, 0.023, 0.019, 0.084, 0.072, ]]

total_number_of_combinations = len(RF_rate_list) * len(num_cells_list)
print('total_number_of_combinations = ' + str(total_number_of_combinations))
cnt = 0

for ind_RF in range(len(RF_rate_list)):

    RF_rc_curr = RF_rate_list[ind_RF][0]
    RF_lc_curr = RF_rate_list[ind_RF][1]
    RF_cr_curr = RF_rate_list[ind_RF][2]
    RF_cl_curr = RF_rate_list[ind_RF][3]
    RF_rl_curr = RF_rate_list[ind_RF][4]
    RF_lr_curr = RF_rate_list[ind_RF][5]

    for num_cells in num_cells_list:
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
        run_name += '_N_' + str(num_cells)

        print('run_name = ' + run_name)

        settings = {}
        # settings['gas_name'] = 'hydrogen'
        # settings['gas_name'] = 'deuterium'
        # settings['gas_name'] = 'tritium'
        settings['gas_name'] = gas_type_list[ind_RF]

        settings = define_default_settings(settings)
        settings['draw_plots'] = False  # plotting not possible on slurm computers without display

        if plasma_mode == 'isoTmfp':
            settings['assume_constant_density'] = True
            settings['assume_constant_temperature'] = True
        elif plasma_mode == 'isoT':
            settings['assume_constant_density'] = False
            settings['assume_constant_temperature'] = True
        elif 'cool' in plasma_mode:
            settings['assume_constant_density'] = False
            settings['assume_constant_temperature'] = False
            settings['plasma_dimension'] = int(plasma_mode.split('d')[-1])

        settings['n0'] = n0
        settings['Ti_0'] = Ti
        settings['Te_0'] = Ti

        settings['cell_size'] = 1.0  # m

        # settings['flux_normalized_termination_cutoff'] = 0.05
        # settings['flux_normalized_termination_cutoff'] = 0.01
        settings['flux_normalized_termination_cutoff'] = 1e-3
        # settings['flux_normalized_termination_cutoff'] = 1e-4

        # for const density right boundary condition
        settings['right_boundary_condition'] = 'none'
        # settings['right_boundary_condition'] = 'adjust_ntL_for_nend'
        # settings['right_boundary_condition'] = 'adjust_ntR_for_nend'
        # settings['right_boundary_condition'] = 'adjust_all_species_for_nend'
        # settings['right_boundary_condition_density_type'] = 'n_expander'
        # settings['n_expander_factor'] = 1e-2
        # settings['n_min'] = n0 * 1e-3
        # settings['time_step_definition_using_species'] = 'only_c_tR'

        settings['number_of_cells'] = num_cells

        # settings['transition_type'] = 'smooth_transition_to_free_flow'
        settings['transition_type'] = 'none'

        settings['Rm'] = 3.0
        # settings['Rm'] = 10.0

        settings['use_RF_terms'] = True
        settings['RF_rc'] = RF_rc_curr
        settings['RF_lc'] = RF_lc_curr
        settings['RF_cr'] = RF_cr_curr
        settings['RF_cl'] = RF_cl_curr
        settings['RF_rl'] = RF_rl_curr
        settings['RF_lr'] = RF_lr_curr

        settings['save_dir'] = main_folder + '/' + run_name
        print('save dir: ' + str(settings['save_dir']))
        os.makedirs(settings['save_dir'], exist_ok=True)
        os.chdir(settings['save_dir'])

        command = rate_eqs_script + ' --settings "' + str(settings) + '"'
        s = Slurm(run_name, slurm_kwargs=slurm_kwargs)
        s.run(command)
        cnt += 1
        print('run # ' + str(cnt) + ' / ' + str(total_number_of_combinations))

        os.chdir(pwd)
