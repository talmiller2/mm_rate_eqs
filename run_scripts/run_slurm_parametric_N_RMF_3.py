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

### RMF (BRF=0.08T)

# set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["1 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.899, 0.888, 0.021, 0.019, 0.051, 0.063, ]]
# set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["1 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.768, 0.774, 0.127, 0.127, 0.107, 0.110, ]]
# set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["2 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.872, 0.821, 0.020, 0.018, 0.063, 0.055, ]]
# set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["2 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.696, 0.714, 0.128, 0.093, 0.147, 0.101, ]]
# set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["3 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.950, 0.421, 0.026, 0.021, 0.010, 0.013, ]]
# set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["3 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.809, 0.413, 0.141, 0.059, 0.050, 0.055, ]]
# set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["4 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.929, 0.267, 0.029, 0.016, 0.020, 0.017, ]]
# set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["4 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.800, 0.274, 0.126, 0.041, 0.036, 0.039, ]]
# set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["5 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.918, 0.217, 0.037, 0.015, 0.010, 0.012, ]]
# set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["5 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.798, 0.213, 0.112, 0.035, 0.027, 0.042, ]]
# set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["6 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.296, 0.283, 0.022, 0.021, 0.040, 0.047, ]]
# set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["6 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.302, 0.316, 0.036, 0.035, 0.049, 0.051, ]]
# set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["7 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.262, 0.925, 0.018, 0.029, 0.016, 0.013, ]]
# set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["7 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.273, 0.813, 0.041, 0.126, 0.056, 0.036, ]]
# set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["8 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.403, 0.201, 0.028, 0.020, 0.034, 0.041, ]]
# set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["8 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.374, 0.176, 0.042, 0.019, 0.035, 0.022, ]]
# set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["9 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.430, 0.145, 0.029, 0.018, 0.036, 0.026, ]]
# set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["9 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.400, 0.149, 0.038, 0.020, 0.028, 0.027, ]]
# set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["10 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.424, 0.128, 0.033, 0.017, 0.029, 0.036, ]]
# set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["10 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.402, 0.128, 0.042, 0.019, 0.030, 0.023, ]]
# set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["11 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.681, 0.162, 0.037, 0.019, 0.046, 0.019, ]]
# set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["11 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.621, 0.148, 0.072, 0.024, 0.021, 0.050, ]]
# set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 1
set_name_list += ["12 (D, iff=1)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.638, 0.668, 0.029, 0.034, 0.095, 0.088, ]]
# set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= deuterium , induced_fields_factor= 0
set_name_list += ["12 (D, iff=0)"]
gas_type_list += ["deuterium"]
RF_rate_list += [[0.580, 0.599, 0.082, 0.078, 0.088, 0.081, ]]

# set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["1 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.500, 0.549, 0.022, 0.020, 0.150, 0.131, ]]
# set 1 , alpha= 1.3 , omega/omega0= 1.559 , beta= 0 , gas= tritium , induced_fields_factor= 0
set_name_list += ["1 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.545, 0.537, 0.071, 0.061, 0.137, 0.158, ]]
# set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
set_name_list += ["2 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.552, 0.314, 0.015, 0.018, 0.091, 0.082, ]]
# set 2 , alpha= 1.48 , omega/omega0= 1.775 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
set_name_list += ["2 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.527, 0.419, 0.071, 0.053, 0.149, 0.143, ]]
# set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["3 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.652, 0.821, 0.018, 0.025, 0.069, 0.073, ]]
# set 3 , alpha= 1.12 , omega/omega0= 1.343 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
set_name_list += ["3 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.588, 0.698, 0.080, 0.099, 0.143, 0.132, ]]
# set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
set_name_list += ["4 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.628, 0.842, 0.017, 0.022, 0.050, 0.065, ]]
# set 4 , alpha= 1.06 , omega/omega0= 1.271 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
set_name_list += ["4 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.649, 0.709, 0.083, 0.112, 0.125, 0.134, ]]
# set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
set_name_list += ["5 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.621, 0.734, 0.019, 0.019, 0.074, 0.078, ]]
# set 5 , alpha= 1.0 , omega/omega0= 1.199 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
set_name_list += ["5 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.634, 0.682, 0.083, 0.156, 0.112, 0.102, ]]
# set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["6 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.886, 0.876, 0.023, 0.027, 0.064, 0.070, ]]
# set 6 , alpha= 0.88 , omega/omega0= 1.055 , beta= 0 , gas= tritium , induced_fields_factor= 0
set_name_list += ["6 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.754, 0.737, 0.132, 0.126, 0.134, 0.138, ]]
# set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 1
set_name_list += ["7 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.826, 0.655, 0.020, 0.018, 0.073, 0.060, ]]
# set 7 , alpha= 1.06 , omega/omega0= 1.271 , beta= 1.4 , gas= tritium , induced_fields_factor= 0
set_name_list += ["7 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.701, 0.623, 0.115, 0.082, 0.153, 0.128, ]]
# set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["8 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.951, 0.426, 0.024, 0.020, 0.017, 0.023, ]]
# set 8 , alpha= 0.76 , omega/omega0= 0.911 , beta= -1.0 , gas= tritium , induced_fields_factor= 0
set_name_list += ["8 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.817, 0.413, 0.149, 0.063, 0.052, 0.051, ]]
# set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 1
set_name_list += ["9 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.934, 0.238, 0.032, 0.015, 0.017, 0.025, ]]
# set 9 , alpha= 0.7 , omega/omega0= 0.839 , beta= -1.4 , gas= tritium , induced_fields_factor= 0
set_name_list += ["9 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.810, 0.257, 0.128, 0.040, 0.044, 0.047, ]]
# set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
set_name_list += ["10 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.894, 0.184, 0.037, 0.013, 0.021, 0.016, ]]
# set 10 , alpha= 0.64 , omega/omega0= 0.767 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
set_name_list += ["10 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.802, 0.196, 0.110, 0.027, 0.024, 0.023, ]]
# set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 1
set_name_list += ["11 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.768, 0.306, 0.039, 0.024, 0.028, 0.034, ]]
# set 11 , alpha= 0.82 , omega/omega0= 0.983 , beta= -1.8 , gas= tritium , induced_fields_factor= 0
set_name_list += ["11 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.723, 0.381, 0.109, 0.057, 0.056, 0.062, ]]
# set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 1
set_name_list += ["12 (T, iff=1)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.763, 0.771, 0.026, 0.030, 0.094, 0.101, ]]
# set 12 , alpha= 1.06 , omega/omega0= 1.271 , beta= 0 , gas= tritium , induced_fields_factor= 0
set_name_list += ["12 (T, iff=0)"]
gas_type_list += ["tritium"]
RF_rate_list += [[0.675, 0.715, 0.087, 0.095, 0.163, 0.140, ]]

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
