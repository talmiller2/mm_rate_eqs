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
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set44_MM_Rm_3_ni_1e21_Ti_10keV_withRMF'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set45_MM_Rm_6_ni_1e21_Ti_10keV_withRMF'
main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set46_MM_Rm_10_ni_1e21_Ti_10keV_withRMF'

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

# based on single_particle calcs: set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T

### RMF (BRF=0.04T)

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

        # settings['Rm'] = 3.0
        # settings['Rm'] = 6.0
        settings['Rm'] = 10.0

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
