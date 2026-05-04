import os
import numpy as np
from scipy.io import savemat, loadmat
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave
import pickle

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/'
main_folder += 'set62_MMM_ni_1e21_Ti_10keV_constmfp'

num_cells_list = [30]
# num_cells_list = [10, 20, 30, 40, 50, 60, 70, 80]

Rm_list = np.arange(2, 10.25, 0.25)
U_list = np.arange(0, 0.95, 0.05)

modes = []
modes += ['mahmir']
modes += ['mekel']

ft_list = []
ft_list += [1]
ft_list += [2]

total_number_of_sets = len(num_cells_list) * len(modes) * len(ft_list)
print('total_number_of_sets = ' + str(total_number_of_sets))

cnt_sets = 0
for num_cells in num_cells_list:
    for mode in modes:
        for ft in ft_list:

            cnt_sets += 1
            print('@@@@@@@ set num', cnt_sets, '/', total_number_of_sets)

            # initialize mats
            flux_mat = np.zeros([len(Rm_list), len(U_list)])

            density_profiles = {}
            population_keys = ['n', 'n_c', 'n_tL', 'n_tR']
            for pop in population_keys:
                density_profiles[pop] = np.zeros([len(Rm_list), len(U_list), num_cells])

            # compile (Rm, U) into matrix
            for ind_Rm, Rm in enumerate(Rm_list):
                for ind_U, U in enumerate(U_list):

                    set_name = 'N_' + str(num_cells)
                    set_name += '_mode_' + str(mode)
                    set_name += f'_ft_{ft:g}'
                    set_name += f'_Rm_{Rm:g}'
                    set_name += f'_U_{U:g}'

                    state_file = main_folder + '/' + set_name + '/state.pickle'
                    if os.path.exists(state_file):
                        try:
                            with open(state_file, 'rb') as fid:
                                state = pickle.load(fid)
                            flux_mat[ind_Rm, ind_U] = state['flux_mean']
                            for pop in population_keys:
                                density_profiles[pop][ind_Rm, ind_U, :] = state[pop]

                        except:
                            print('failed to load', state_file)
                    else:
                        print(state_file, 'doesnt exist.')

            # save data
            save_mat_dict = {}
            save_mat_dict['Rm_list'] = Rm_list
            save_mat_dict['U_list'] = U_list
            save_mat_dict['flux_mat'] = flux_mat
            for pop in population_keys:
                save_mat_dict[pop] = density_profiles[pop]

            compiled_set_name = 'compiled_'
            compiled_set_name += 'N_' + str(num_cells)
            compiled_set_name += '_mode_' + str(mode)
            compiled_set_name += f'_ft_{ft:g}'
            compiled_save_file = main_folder + '/' + compiled_set_name + '.mat'
            print('saving', compiled_save_file)
            savemat(compiled_save_file, save_mat_dict)
