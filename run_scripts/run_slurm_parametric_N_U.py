import os

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set1/'

mode = 'isoTmfp'
# mode = 'isoT'
# mode = 'cool'

N_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

for num_cells in N_list:
    for U in U_list:
        run_name = mode + '_N_' + str(num_cells) + '_U_' + str(U)
        print('run_name = ' + run_name)

        settings = {}
        settings = define_default_settings(settings)
        settings['draw_plots'] = False  # plotting not possible on slurm computers without display

        if mode == 'isoTmfp':
            settings['assume_constant_density'] = True
            settings['assume_constant_temperature'] = True
        elif mode == 'isoT':
            settings['assume_constant_density'] = False
            settings['assume_constant_temperature'] = True
        elif mode == 'cool':
            settings['assume_constant_density'] = False
            settings['assume_constant_temperature'] = True

        settings['save_dir'] = main_folder + '/' + run_name
        print('save dir: ' + str(settings['save_dir']))
        os.makedirs(settings['save_dir'], exist_ok=True)
        os.chdir(settings['save_dir'])

        command = rate_eqs_script + ' --settings "' + str(settings) + '"'
        s = Slurm(run_name)
        s.run(command)

        os.chdir(pwd)
