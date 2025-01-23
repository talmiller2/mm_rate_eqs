import os
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
# main_folder += 'set47_MM_Rm_10_ni_1e21_Ti_10keV_withRMF'
# main_folder += 'set48_MM_Rm_10_ni_1e21_Ti_10keV_withRMF_zeroRL_fluxeps1e-2'
# main_folder += 'set49_MM_Rm_10_ni_1e21_Ti_10keV_withRMF_fluxeps1e-2'
# main_folder += 'set50_MM_Rm_10_ni_1e20_Ti_10keV_withRMF_zeroRL_fluxeps1e-2'
main_folder += 'set54_MM_Rm_10_ni_1e20_Ti_10keV_smooth_fluxeps1e-3'

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
# num_cells_list = [50]

# load single_particle compiled mat
# single_particle_dir = '/Users/talmiller/Downloads/single_particle/'
single_particle_dir = '/home/talm/code/single_particle/slurm_runs/'
# single_particle_dir += '/set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
single_particle_dir += '/set54_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'

# extract variables from saved single particle calcs
settings_file = single_particle_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings = pickle.load(fid)
l = settings['l']

field_dict_file = single_particle_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict = pickle.load(fid)
Rm = field_dict['Rm']

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

        # set_name = 'compiled_'
        set_name = 'smooth_compiled_'
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

        total_number_of_combinations = len(num_cells_list) * len(alpha_loop_list) * len(beta_loop_list)
        print('total_number_of_combinations = ' + str(total_number_of_combinations))
        cnt = 0

        # create corresponding dir in the mm_rate_eqs
        sub_folder = main_folder + '/' + set_name
        print('creating sub_folder:', sub_folder)
        os.makedirs(sub_folder, exist_ok=True)

        for ind_beta, beta in enumerate(beta_loop_list):
            for ind_alpha, alpha in enumerate(alpha_loop_list):

                RF_rc_curr = RF_rates_mat_dict['N_rc'][ind_beta, ind_alpha]
                RF_lc_curr = RF_rates_mat_dict['N_lc'][ind_beta, ind_alpha]
                RF_cr_curr = RF_rates_mat_dict['N_cr'][ind_beta, ind_alpha]
                RF_cl_curr = RF_rates_mat_dict['N_cl'][ind_beta, ind_alpha]
                RF_rl_curr = RF_rates_mat_dict['N_rl'][ind_beta, ind_alpha]
                RF_lr_curr = RF_rates_mat_dict['N_lr'][ind_beta, ind_alpha]
                # RF_rl_curr = 0
                # RF_lr_curr = 0

                for num_cells in num_cells_list:
                    run_name = plasma_mode
                    run_name += '_' + gas_name
                    RF_label = 'alpha_' + str(alpha) + '_beta_' + str(beta)
                    run_name += '_' + RF_label
                    run_name += '_N_' + str(num_cells)

                    print('run_name = ' + run_name)

                    settings = {}
                    settings['gas_name'] = gas_name

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
                        settings['assume_constant_temperature'] = False
                        settings['plasma_dimension'] = int(plasma_mode.split('d')[-1])

                    settings['n0'] = n0
                    settings['Ti_0'] = Ti
                    settings['Te_0'] = Ti

                    # settings['cell_size'] = 1.0  # m
                    settings['cell_size'] = l

                    # settings['flux_normalized_termination_cutoff'] = 0.05
                    # settings['flux_normalized_termination_cutoff'] = 1e-2
                    settings['flux_normalized_termination_cutoff'] = 1e-3
                    # settings['flux_normalized_termination_cutoff'] = 1e-4

                    # for const density right boundary condition
                    settings['right_boundary_condition'] = 'none'
                    # settings['right_boundary_condition'] = 'adjust_ntL_for_nend'
                    # settings['right_boundary_condition'] = 'adjust_ntR_for_nend'
                    # settings['right_boundary_condition'] = 'adjust_all_species_for_nend'
                    # settings['right_boundary_condition_density_type'] = 'n_expander'
                    # settings['n_expander_factor'] = 1e-2
                    # settings['n_min'] = n0 * 1e-3
                    # settings['time_step_definition_using_species'] = 'only_c_tR'

                    settings['number_of_cells'] = num_cells

                    # settings['transition_type'] = 'smooth_transition_to_free_flow'
                    settings['transition_type'] = 'none'

                    # settings['Rm'] = 3.0
                    # settings['Rm'] = 6.0
                    # settings['Rm'] = 10.0
                    settings['Rm'] = Rm

                    settings['use_RF_terms'] = True
                    settings['RF_rc'] = RF_rc_curr
                    settings['RF_lc'] = RF_lc_curr
                    settings['RF_cr'] = RF_cr_curr
                    settings['RF_cl'] = RF_cl_curr
                    settings['RF_rl'] = RF_rl_curr
                    settings['RF_lr'] = RF_lr_curr

                    settings['save_dir'] = sub_folder + '/' + run_name
                    print('save dir: ' + str(settings['save_dir']))

                    state_file = settings['save_dir'] + '/state.pickle'
                    if os.path.exists(state_file):
                        print('exists, skipping.')
                    else:
                        os.makedirs(settings['save_dir'], exist_ok=True)
                        os.chdir(settings['save_dir'])

                        command = rate_eqs_script + ' --settings "' + str(settings) + '"'
                        s = Slurm(run_name, slurm_kwargs=slurm_kwargs)
                        s.run(command)
                        cnt += 1
                        print('run # ' + str(cnt) + ' / ' + str(total_number_of_combinations))

                        os.chdir(pwd)
