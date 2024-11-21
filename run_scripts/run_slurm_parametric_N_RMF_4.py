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
main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set45_MM_Rm_6_ni_1e21_Ti_10keV_withRMF'

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

# based on single_particle calcs: set51_B0_1T_l_1m_Post_Rm_6_intervals_D_T

### RMF (BRF=0.04T)

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
        settings['Rm'] = 6.0

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
