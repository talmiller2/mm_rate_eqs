from slurmpy.slurmpy import Slurm
import os
from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

rate_eqs_script = get_script_rate_eqs_slave()

settings = {}
settings = define_default_settings(settings)

settings['save_dir'] = '~/code/mm_rate_eqs/runs/slurm_runs/test/'
print('save dir: ' + str(settings['save_dir']))

os.makedirs(settings['save_dir'], exist_ok=True)

command = rate_eqs_script + ' --settings "' + str(settings) + '"'
s = Slurm('job_name')
s.run(command)
