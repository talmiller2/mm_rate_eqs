import os

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set2_Rm_3/'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set6_Rm_3_mfp_over_cell_1_mfp_limitX100/'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set11_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2/'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set12_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2_rbc_adjut_ntL_timestepdef_without_ntL/'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set13_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2_rbc_adjut_ntR/'
# n0 = 3.875e22  # m^-3

# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set4_Rm_3_mfp_over_cell_4/'
# n0 = 1e22  # m^-3

# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set5_Rm_3_mfp_over_cell_20/'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set7_Rm_3_mfp_over_cell_20_mfp_limitX100/'
# n0 = 2e21  # m^-3

# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set10_Rm_3_mfp_over_cell_0.04_mfp_limitX100/'
# n0 = 1e24  # m^-3

# n0 = 2e22  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set14_MM_Rm_3_ni_2e22'

# n0 = 2e22  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set15_MM_Rm_3_ni_2e22_nend_1e-2_rbc_adjust_ntR'

# n0 = 4e23  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set16_MM_Rm_3_ni_4e23'

# n0 = 1e21  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set17_MM_Rm_3_ni_1e21'

############

# n0 = 2e22  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set20_MM_Rm_3_ni_2e22_trans_type_none'

# n0 = 2e22  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set21_MM_Rm_3_ni_2e22_trans_type_none_trans_fac_1'

# n0 = 1e21  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set22_MM_Rm_3_ni_1e21_trans_type_none'

# n0 = 2e20  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set24_MM_Rm_3_ni_2e20_trans_type_none'

# n0 = 4e23  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set25_MM_Rm_3_ni_4e23_trans_type_none'

# n0 = 2e20  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set26_MM_Rm_3_ni_2e20_trans_type_none_flux_cutoff_0.01'

# n0 = 2e22  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set27_MM_Rm_3_ni_2e22_trans_type_none_flux_cutoff_1e-3'

# n0 = 2e22  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set28_MM_Rm_3_ni_2e22_trans_type_none_flux_cutoff_1e-4'

n0 = 2e20  # m^-3
main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set29_MM_Rm_3_ni_2e20_trans_type_none_flux_cutoff_1e-4'

# n0 = 4e23  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set30_MM_Rm_3_ni_4e23_trans_type_none_flux_cutoff_1e-4'


slurm_kwargs = {'partition': 'core'}  # default
# slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
plasma_modes += ['coold1']
plasma_modes += ['coold2']
plasma_modes += ['coold3']
# plasma_modes += ['coold']
# plasma_modes += ['cool_mfpcutoff']

LC_modes = []
LC_modes += ['sLC']  # static loss cone
# LC_modes += ['dLC']  # dynamic loss cone

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# num_cells_list = [3, 5, 8, 10, 15, 30, 50, 70, 100]
# num_cells_list = [3, 5, 8, 10, 15, 20, 30, 50, 70, 100, 130, 150]
# U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
# num_cells_list = [50, 70, 100, 130, 150]
U_list = [0]

total_number_of_combinations = len(plasma_modes) * len(LC_modes) * len(num_cells_list) * len(U_list)
print('total_number_of_combinations = ' + str(total_number_of_combinations))
cnt = 0

for plasma_mode in plasma_modes:
    for LC_mode in LC_modes:
        for num_cells in num_cells_list:
            for U in U_list:
                run_name = plasma_mode
                run_name += '_N_' + str(num_cells) + '_U_' + str(U)
                run_name += '_' + LC_mode

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
                    settings['assume_constant_density'] = False
                    settings['plasma_dimension'] = int(plasma_mode.split('d')[-1])

                settings['n0'] = n0

                # settings['flux_normalized_termination_cutoff'] = 0.05
                # settings['flux_normalized_termination_cutoff'] = 0.01
                # settings['flux_normalized_termination_cutoff'] = 1e-3
                settings['flux_normalized_termination_cutoff'] = 1e-4

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

                settings['U0'] = U
                if LC_mode == 'sLC':
                    settings['alpha_definition'] = 'geometric_constant'
                elif LC_mode == 'dLC':
                    settings['alpha_definition'] = 'geometric_local'

                # settings['transition_type'] = 'smooth_transition_to_free_flow'
                settings['transition_type'] = 'none'

                settings['transmission_factor'] = 2.0
                # settings['transmission_factor'] = 1.0

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
