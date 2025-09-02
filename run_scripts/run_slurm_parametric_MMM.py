import os
import numpy as np
from scipy.io import loadmat
import pickle

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

n0 = 1e21  # m^-3
# n0 = 1e20  # m^-3
Ti = 10 * 1e3  # eV

main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/'
# main_folder += 'set57_MMM_ni_1e21_Ti_10keV_constmfp'
main_folder += 'set58_MMM_ni_1e21_Ti_10keV_constmfp_trfix'

slurm_kwargs = {}
slurm_kwargs['partition'] = 'core'
# slurm_kwargs['partition'] = 'testCore'
# slurm_kwargs['partition'] = 'socket'
# slurm_kwargs['partition'] = 'testSocket'
slurm_kwargs['ntasks'] = 1
slurm_kwargs['cpus-per-task'] = 1

# gas_name = 'deuterium'
gas_name = 'tritium'

# num_cells_list = [10, 30, 50, 80]
num_cells_list = [20, 40, 60, 70]
# scat_factor_list = [0.1, 1]
# scat_asym_list = [0.5, 1, 2]
scat_factor_list = [1]
scat_asym_list = [1]
Rm_list = np.round(np.linspace(1.1, 10, 21), 2)
U_list = np.round(np.linspace(0, 1, 21), 2)

total_number_of_sets = len(num_cells_list) * len(scat_factor_list) * len(scat_asym_list) * len(Rm_list) * len(U_list)
print('total_number_of_sets = ' + str(total_number_of_sets))
cnt_sets = 0

for num_cells in num_cells_list:
    for scat_factor in scat_factor_list:
        for scat_asym_factor in scat_asym_list:
            for Rm in Rm_list:
                for U in U_list:

                    cnt_sets += 1

                    set_name = 'N_' + str(num_cells)
                    set_name += '_scatfac_' + str(scat_factor)
                    set_name += '_scatasym_' + str(scat_asym_factor)
                    set_name += '_Rm_' + str(Rm)
                    set_name += '_U_' + str(U)
                    set_name += '_' + gas_name
                    # print(set_name)

                    print('@@@@@@@ set num', cnt_sets, '/', total_number_of_sets, ':', set_name)

                    settings = {}
                    settings['gas_name'] = gas_name

                    settings = define_default_settings(settings)
                    settings['draw_plots'] = False  # plotting not possible on slurm computers without display

                    settings['n0'] = n0
                    settings['Ti_0'] = Ti
                    settings['Te_0'] = Ti

                    settings['cell_size'] = 1.0  # m

                    # MMM
                    settings['U0'] = U
                    settings['alpha_definition'] = 'geometric_constant_U0'

                    # mfp
                    settings['assume_constant_density'] = True
                    settings['assume_constant_temperature'] = True
                    settings['ion_scattering_rate_factor'] = 1800 * scat_factor
                    settings['scat_factor_right'] = scat_asym_factor
                    settings['scat_factor_left'] = 1 / scat_asym_factor
                    settings['energy_conservation_scheme'] = 'MMM'

                    # settings['flux_normalized_termination_cutoff'] = 0.05
                    # settings['flux_normalized_termination_cutoff'] = 1e-2
                    settings['flux_normalized_termination_cutoff'] = 1e-3
                    # settings['flux_normalized_termination_cutoff'] = 1e-4

                    settings['right_boundary_condition'] = 'none'

                    settings['number_of_cells'] = num_cells

                    # settings['transition_type'] = 'smooth_transition_to_free_flow'
                    settings['transition_type'] = 'none'

                    settings['Rm'] = Rm

                    settings['save_dir'] = main_folder + '/' + set_name
                    state_file = settings['save_dir'] + '/state.pickle'
                    if os.path.exists(state_file):
                        print('exists, skipping.')
                    else:
                        os.makedirs(settings['save_dir'], exist_ok=True)
                        os.chdir(settings['save_dir'])

                        command = rate_eqs_script + ' --settings "' + str(settings) + '"'
                        s = Slurm(set_name, slurm_kwargs=slurm_kwargs)
                        s.run(command)
