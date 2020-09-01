from slurmpy.slurmpy import Slurm

from default_settings import define_default_settings
from slurm_functions import get_rate_equations_steady_state_slave_script

slave_script = get_rate_equations_steady_state_slave_script()

settings = {}
settings = define_default_settings(settings)

settings['save_dir'] = 'runs/test_slurm/'
print('save dir: ' + str(settings['save_dir']))

command = slave_script + ' --settings ' + str(settings)
s = Slurm('job_name')
s.run(command)
