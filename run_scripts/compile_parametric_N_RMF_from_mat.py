import os
import numpy as np
from scipy.io import savemat, loadmat
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave
import pickle

pwd = os.getcwd()
rate_eqs_script = get_script_rate_eqs_slave()

n0 = 1e21  # m^-3
Ti = 10 * 1e3  # eV
main_folder = '/home/talm/code/mm_rate_eqs/runs/slurm_runs/set47_MM_Rm_10_ni_1e21_Ti_10keV_withRMF'

slurm_kwargs = {}
slurm_kwargs['partition'] = 'core'
# slurm_kwargs['partition'] = 'testCore'
# slurm_kwargs['partition'] = 'socket'
# slurm_kwargs['partition'] = 'testSocket'
slurm_kwargs['ntasks'] = 1
slurm_kwargs['cpus-per-task'] = 1

plasma_mode = 'isoT'

num_cells_list = [10, 30, 50]
# num_cells_list = [30]

# load single_particle compiled mat
# single_particle_dir = '/Users/talmiller/Downloads/single_particle/'
single_particle_dir = '/home/talm/code/single_particle/slurm_runs/'
single_particle_dir += '/set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'

gas_name_list = ['deuterium', 'tritium']

RF_type_list = []
RF_amplitude_list = []
induced_fields_factor_list = []
with_kr_correction_list = []

RF_type_list += ['electric_transverse']
RF_amplitude_list += [25]  # kV/m
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['electric_transverse']
RF_amplitude_list += [50]  # kV/m
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.02]  # T
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.02]  # T
induced_fields_factor_list += [0]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.04]  # T
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.04]  # T
induced_fields_factor_list += [0]
with_kr_correction_list += [True]

total_number_of_sets = len(RF_type_list) * len(gas_name_list)
print('total_number_of_sets = ' + str(total_number_of_sets))
cnt_sets = 0

for RF_type, RF_amplitude, induced_fields_factor, with_kr_correction \
        in zip(RF_type_list, RF_amplitude_list, induced_fields_factor_list, with_kr_correction_list):
    for gas_name in gas_name_list:

        cnt_sets += 1
        print('@@@@@@@ set num', cnt_sets, '/', total_number_of_sets)

        time_step_tau_cyclotron_divisions = 50
        # time_step_tau_cyclotron_divisions = 100
        # sigma_r0 = 0
        sigma_r0 = 0.05
        # sigma_r0 = 0.1
        radial_distribution = 'uniform'

        # theta_type = 'sign_vz0'
        theta_type = 'sign_vz'

        if gas_name == 'deuterium':
            gas_name_short = 'D'
        elif gas_name == 'tritium':
            gas_name_short = 'T'
        else:
            gas_name_short = 'DTmix'

        set_name = 'compiled_'
        set_name += theta_type + '_'
        if RF_type == 'electric_transverse':
            set_name += 'ERF_' + str(RF_amplitude)
        elif RF_type == 'magnetic_transverse':
            set_name += 'BRF_' + str(RF_amplitude)
        if induced_fields_factor < 1.0:
            set_name += '_iff' + str(induced_fields_factor)
        if with_kr_correction == True:
            set_name += '_withkrcor'
        set_name += '_tcycdivs' + str(time_step_tau_cyclotron_divisions)
        if sigma_r0 > 0:
            set_name += '_sigmar' + str(sigma_r0)
            if radial_distribution == 'normal':
                set_name += 'norm'
            elif radial_distribution == 'uniform':
                set_name += 'unif'
        set_name += '_' + gas_name
        print(set_name)
        single_particle_file = single_particle_dir + '/' + set_name + '.mat'

        RF_rates_mat_dict = loadmat(single_particle_file)
        alpha_loop_list = RF_rates_mat_dict['alpha_loop_list'][0]
        beta_loop_list = RF_rates_mat_dict['beta_loop_list'][0]
        flux_mat = np.nan * RF_rates_mat_dict['N_rc']

        total_number_of_combinations = len(num_cells_list) * len(alpha_loop_list) * len(beta_loop_list)
        print('total_number_of_combinations = ' + str(total_number_of_combinations))
        cnt = 0

        sub_folder = main_folder + '/' + set_name

        for num_cells in num_cells_list:
            for ind_beta, beta in enumerate(beta_loop_list):
                for ind_alpha, alpha in enumerate(alpha_loop_list):
                    cnt += 1

                    run_name = plasma_mode
                    run_name += '_' + gas_name
                    RF_label = 'alpha_' + str(alpha) + '_beta_' + str(beta)
                    run_name += '_' + RF_label
                    run_name += '_N_' + str(num_cells)

                    state_file = sub_folder + '/' + run_name + '/state.pickle'

                    if os.path.exists(state_file):
                        print('loading state of run # ' + str(cnt) + ' / ' + str(total_number_of_combinations))
                        try:
                            with open(state_file, 'rb') as fid:
                                state = pickle.load(fid)
                            flux_mat[ind_beta, ind_alpha] = state['flux_mean']
                        except:
                            print('FAILED TO LOAD.')
                    else:
                        print(state_file, 'doesnt exist.')

            # save data
            save_mat_dict = {}
            save_mat_dict['alpha_loop_list'] = alpha_loop_list
            save_mat_dict['beta_loop_list'] = beta_loop_list
            save_mat_dict['flux_mat'] = flux_mat
            compiled_save_file = main_folder + '/compiled_' + set_name + '_N_' + str(num_cells) + '.mat'
            print('saving', compiled_save_file)
            savemat(compiled_save_file, save_mat_dict)
