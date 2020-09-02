import os

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set1/'

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
plasma_modes += ['cool']

LC_modes = []
LC_modes += ['sLC']  # static loss cone
LC_modes += ['dLC']  # dynamic loss cone

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

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
                elif plasma_mode == 'cool':
                    settings['assume_constant_density'] = False
                    settings['assume_constant_temperature'] = True

                if LC_mode == 'sLC':
                    settings['alpha_definition'] = 'geometric_constant'
                elif LC_mode == 'dLC':
                    settings['alpha_definition'] = 'geometric_local'

                settings['save_dir'] = main_folder + '/' + run_name
                print('save dir: ' + str(settings['save_dir']))
                os.makedirs(settings['save_dir'], exist_ok=True)
                os.chdir(settings['save_dir'])

                command = rate_eqs_script + ' --settings "' + str(settings) + '"'
                s = Slurm(run_name)
                s.run(command)
                cnt += 1
                print('run # ' + str(cnt) + ' / ' + str(total_number_of_combinations))

                os.chdir(pwd)
