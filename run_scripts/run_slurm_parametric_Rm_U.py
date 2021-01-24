import os

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set3_N_20/'
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set8_N_30_mfp_over_cell_1_mfp_limitX100/'
# n0 = 3.875e22  # m^-3

# n0 = 1e22  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set9_N_30_mfp_over_cell_40_mfp_limitX100/'

# n0 = 2e22  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set18_MM_N_30_ni_2e22'

# n0 = 1e21  # m^-3
# main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set19_MM_N_30_ni_1e21'

###########

n0 = 2e22  # m^-3
main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set23_MM_N_30_ni_2e22_trans_type_none'

# slurm_kwargs = {'partition': 'core'}  # default
slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
plasma_modes += ['coold1']
plasma_modes += ['coold2']
plasma_modes += ['coold3']

LC_modes = []
LC_modes += ['sLC']  # static loss cone
# LC_modes += ['dLC']  # dynamic loss cone

# num_cells = 20
# Rm_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

num_cells = 30
# Rm_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
Rm_list = [1.1, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0]
# U_list = [0, 0.05, 0.1, 0.3, 0.5]
U_list = [0]

total_number_of_combinations = len(plasma_modes) * len(LC_modes) * len(Rm_list) * len(U_list)
print('total_number_of_combinations = ' + str(total_number_of_combinations))
cnt = 0

for plasma_mode in plasma_modes:
    for LC_mode in LC_modes:
        for Rm in Rm_list:
            for U in U_list:
                run_name = plasma_mode
                run_name += '_Rm_' + str(Rm) + '_U_' + str(U)
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
                settings['number_of_cells'] = num_cells
                settings['Rm'] = Rm

                settings['U0'] = U
                if LC_mode == 'sLC':
                    settings['alpha_definition'] = 'geometric_constant'
                elif LC_mode == 'dLC':
                    settings['alpha_definition'] = 'geometric_local'

                # if plasma_mode == 'cool_mfpcutoff':
                #     settings['transition_type'] = 'smooth_transition_to_free_flow'
                # settings['transition_type'] = 'smooth_transition_to_free_flow'
                settings['transition_type'] = 'none'

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
