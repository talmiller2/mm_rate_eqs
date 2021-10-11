import os

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

slurm_kwargs = {'partition': 'core'}  # default
# slurm_kwargs = {'partition': 'socket'}
# slurm_kwargs = {'partition': 'testing'}

main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set36_RFscat_ni_1e20_T_10keV_N_20'

nu_RF_c_list = [0.05, 0.1, 0.5, 0.05, 0.1, 0.5, 0.05, 0.1, 0.5]
nu_RF_tL_list = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3]
nu_RF_tR_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

if len(nu_RF_c_list) != len(nu_RF_tL_list) or len(nu_RF_c_list) != len(nu_RF_tR_list) \
        or len(nu_RF_tL_list) != len(nu_RF_tR_list):
    raise ValueError('lengths incompatible.')

nu_RF_factor_list = [0.1, 0.2, 0.5, 0.8, 1, 2, 5, 10, 15, 20, 25, 30]

total_number_of_combinations = len(nu_RF_factor_list) * len(nu_RF_c_list)
print('total_number_of_combinations = ' + str(total_number_of_combinations))

cnt = 0
for nu_RF_factor in nu_RF_factor_list:
    for nu_RF_c, nu_RF_tL, nu_RF_tR in zip(nu_RF_c_list, nu_RF_tL_list, nu_RF_tR_list):
        run_name = ''
        run_name += 'nu_RF_c_' + str(nu_RF_c)
        run_name += '_tL_' + str(nu_RF_tL)
        run_name += '_tR_' + str(nu_RF_tR)
        run_name += '_fac_' + str(nu_RF_factor)

        print('run_name = ' + run_name)

        settings = {}

        settings['draw_plots'] = False  # plotting not possible on slurm computers without display
        settings['assume_constant_density'] = False
        settings['assume_constant_temperature'] = True

        settings['n0'] = 1e20  # m^-3
        settings['Ti_0'] = 10 * 1e3  # eV
        settings['Te_0'] = 10 * 1e3  # eV
        settings['number_of_cells'] = 20
        settings['Rm'] = 3.0

        settings['flux_normalized_termination_cutoff'] = 1e-3

        settings['use_effective_RF_scattering'] = True
        settings['nu_RF_c'] = nu_RF_c * nu_RF_factor
        settings['nu_RF_tL'] = nu_RF_tL * nu_RF_factor
        settings['nu_RF_tR'] = nu_RF_tR * nu_RF_factor

        settings['save_dir'] = main_folder + '/' + run_name
        print('save dir: ' + str(settings['save_dir']))
        os.makedirs(settings['save_dir'], exist_ok=True)
        os.chdir(settings['save_dir'])

        settings = define_default_settings(settings)

        command = rate_eqs_script + ' --settings "' + str(settings) + '"'
        s = Slurm(run_name, slurm_kwargs=slurm_kwargs)
        s.run(command)
        cnt += 1
        print('run # ' + str(cnt) + ' / ' + str(total_number_of_combinations))

        os.chdir(pwd)
