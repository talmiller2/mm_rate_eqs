from slurmpy.slurmpy import Slurm
import os
from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

settings = {}
settings = define_default_settings(settings)

settings['save_dir'] = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/test/'
print('save dir: ' + str(settings['save_dir']))
os.makedirs(settings['save_dir'], exist_ok=True)
os.chdir(settings['save_dir'])

settings['draw_plots'] = False  # plotting not possible on slurm computers without display

command = rate_eqs_script + ' --settings "' + str(settings) + '"'
s = Slurm('test_slurm_run')
s.run(command)

os.chdir(pwd)
