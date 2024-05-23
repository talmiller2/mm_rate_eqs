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
main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set43_MM_Rm_3_ni_1e21_Ti_10keV_withRMF'

slurm_kwargs = {}
slurm_kwargs['partition'] = 'core'
# slurm_kwargs['partition'] = 'socket'
# slurm_kwargs['partition'] = 'testSocket'
slurm_kwargs['ntasks'] = 1
slurm_kwargs['cpus-per-task'] = 1

plasma_mode = 'isoT'

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

###########################
set_name_list = []
gas_type_list = []
RF_capacity_rc_list = []
RF_capacity_lc_list = []
RF_capacity_cr_list = []
RF_capacity_cl_list = []

######################

# based on single_particle: set45_B0_1T_l_1m_Post_Rm_3_intervals_D_T: compiled_BRF_0.04_iff0_tcycdivs40_sigmar0.1_deuterium/tritium
# set1, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.839, k/2pi=-2
set_name_list += ['1 (D)']
gas_type_list += ['deuterium']
RF_capacity_rc_list += [0.375]
RF_capacity_lc_list += [0.078]
RF_capacity_cr_list += [0.029]
RF_capacity_cl_list += [0.021]
set_name_list += ['1 (T)']
gas_type_list += ['tritium']
RF_capacity_rc_list += [0.661]
RF_capacity_lc_list += [0.091]
RF_capacity_cr_list += [0.074]
RF_capacity_cl_list += [0.045]

# set2, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.983, k/2pi=-2
set_name_list += ['2 (D)']
gas_type_list += ['deuterium']
RF_capacity_rc_list += [0.558]
RF_capacity_lc_list += [0.090]
RF_capacity_cr_list += [0.037]
RF_capacity_cl_list += [0.036]
set_name_list += ['2 (T)']
gas_type_list += ['tritium']
RF_capacity_rc_list += [0.470]
RF_capacity_lc_list += [0.174]
RF_capacity_cr_list += [0.052]
RF_capacity_cl_list += [0.047]

# set3, Rm=3, l=1m, BRF=0.04T, omega/omega0T=1.175, k/2pi=-1.6
set_name_list += ['3 (D)']
gas_type_list += ['deuterium']
RF_capacity_rc_list += [0.775]
RF_capacity_lc_list += [0.094]
RF_capacity_cr_list += [0.045]
RF_capacity_cl_list += [0.069]
set_name_list += ['3 (T)']
gas_type_list += ['tritium']
RF_capacity_rc_list += [0.293]
RF_capacity_lc_list += [0.625]
RF_capacity_cr_list += [0.058]
RF_capacity_cl_list += [0.075]

# set4, Rm=3, l=1m, BRF=0.04T, omega/omega0T=0.720, k/2pi=-2
set_name_list += ['4 (D)']
gas_type_list += ['deuterium']
RF_capacity_rc_list += [0.210]
RF_capacity_lc_list += [0.066]
RF_capacity_cr_list += [0.014]
RF_capacity_cl_list += [0.013]
set_name_list += ['4 (T)']
gas_type_list += ['tritium']
RF_capacity_rc_list += [0.345]
RF_capacity_lc_list += [0.094]
RF_capacity_cr_list += [0.022]
RF_capacity_cl_list += [0.031]

#######################


total_number_of_combinations = len(RF_capacity_cl_list) * len(num_cells_list)
print('total_number_of_combinations = ' + str(total_number_of_combinations))
cnt = 0

for ind_RF in range(len(RF_capacity_cl_list)):

    for num_cells in num_cells_list:
        run_name = plasma_mode
        run_name += '_' + gas_type_list[ind_RF]
        RF_label = 'RF_terms' \
                   + '_cl_' + str(RF_capacity_cl_list[ind_RF]) \
                   + '_cr_' + str(RF_capacity_cr_list[ind_RF]) \
                   + '_lc_' + str(RF_capacity_lc_list[ind_RF]) \
                   + '_rc_' + str(RF_capacity_rc_list[ind_RF])
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
