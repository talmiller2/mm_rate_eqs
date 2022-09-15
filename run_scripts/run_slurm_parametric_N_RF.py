import os

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

n0 = 1e21  # m^-3
main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set36_MM_Rm_3_ni_1e21_Ti_10keV_withRF'

slurm_kwargs = {'partition': 'core'}  # default
# slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

plasma_mode = 'isoT'

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

RF_capacity_cl_list = []
RF_capacity_cr_list = []
RF_capacity_lc_list = []
RF_capacity_rc_list = []

# Rm=3, l=1m, ERF=50kV/m, alpha=1, beta=0, selectivity=1
RF_capacity_cl_list += [0.026]
RF_capacity_cr_list += [0.026]
RF_capacity_lc_list += [0.61]
RF_capacity_rc_list += [0.61]

# Rm=3, l=1m, ERF=50kV/m, alpha=1, beta=-1, selectivity=1.46
RF_capacity_cl_list += [0.02]
RF_capacity_cr_list += [0.031]
RF_capacity_lc_list += [0.664]
RF_capacity_rc_list += [0.455]

# Rm=3, l=1m, ERF=50kV/m, alpha=0.9, beta=-5, selectivity=3.31
RF_capacity_cl_list += [0.018]
RF_capacity_cr_list += [0.01]
RF_capacity_lc_list += [0.296]
RF_capacity_rc_list += [0.089]

# Rm=3, l=1m, ERF=50kV/m, alpha=0.8, beta=-5, selectivity=6.52
RF_capacity_cl_list += [0.023]
RF_capacity_cr_list += [0.008]
RF_capacity_lc_list += [0.484]
RF_capacity_rc_list += [0.074]

total_number_of_combinations = len(RF_capacity_cl_list) * len(num_cells_list)
print('total_number_of_combinations = ' + str(total_number_of_combinations))
cnt = 0

for ind_RF in range(len(RF_capacity_cl_list)):

    for num_cells in num_cells_list:
        run_name = plasma_mode

        run_name += '_RF_terms_' + 'cl_' + str(RF_capacity_cl_list[ind_RF]) \
                    + '_cr_' + str(RF_capacity_cr_list[ind_RF]) \
                    + '_lc_' + str(RF_capacity_lc_list[ind_RF]) \
                    + '_rc_' + str(RF_capacity_rc_list[ind_RF])
        run_name += '_N_' + str(num_cells)

        print('run_name = ' + run_name)

        settings = {}
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

        settings['RF_capacity_cl'] = RF_capacity_cl_list[ind_RF]
        settings['RF_capacity_cr'] = RF_capacity_cr_list[ind_RF]
        settings['RF_capacity_lc'] = RF_capacity_lc_list[ind_RF]
        settings['RF_capacity_rc'] = RF_capacity_rc_list[ind_RF]

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
