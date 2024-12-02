import os
from scipy.io import loadmat

from slurmpy.slurmpy import Slurm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.slurm_functions import get_script_rate_eqs_slave

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

# num_cells_list = [10, 30, 50]
num_cells_list = [30]

# load single_particle compiled mat
# single_particle_dir = '/Users/talmiller/Downloads/single_particle/'
single_particle_dir = '/home/talm/code/single_particle/slurm_runs/'
single_particle_dir += '/set53_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'

gas_name_list = ['deuterium', 'tritium']

gas_name = 'deuterium'
# gas_name = 'DT_mix'
# gas_name = 'tritium'

for gas_name in gas_name_list:

    induced_fields_factor = 1
    # induced_fields_factor = 0
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

    # with_kr_correction = False
    with_kr_correction = True

    RF_type = 'electric_transverse'
    E_RF_kVm = 25  # kV/m
    # E_RF_kVm = 50  # kV/m
    # E_RF_kVm = 100  # kV/m

    # RF_type = 'magnetic_transverse'
    # B_RF = 0.02  # T
    # B_RF = 0.04  # T
    # B_RF = 0.08  # T

    set_name = 'compiled_'
    set_name += theta_type + '_'
    if RF_type == 'electric_transverse':
        set_name += 'ERF_' + str(E_RF_kVm)
    elif RF_type == 'magnetic_transverse':
        set_name += 'BRF_' + str(B_RF)
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
    N_rc_1 = RF_rates_mat_dict['N_rc']
    N_lc_1 = RF_rates_mat_dict['N_lc']
    N_cr_1 = RF_rates_mat_dict['N_cr']
    N_cl_1 = RF_rates_mat_dict['N_cl']
    N_rl_1 = RF_rates_mat_dict['N_rl']
    N_lr_1 = RF_rates_mat_dict['N_lr']

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

            for num_cells in num_cells_list:
                run_name = plasma_mode
                run_name += '_' + gas_name
                # RF_label = 'RF_terms' \
                #            + '_rc_' + str(RF_rc_curr) \
                #            + '_lc_' + str(RF_lc_curr) \
                #            + '_cr_' + str(RF_cr_curr) \
                #            + '_cl_' + str(RF_cl_curr) \
                #            + '_rl_' + str(RF_rl_curr) \
                #            + '_lr_' + str(RF_lr_curr)
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

                settings['cell_size'] = 1.0  # m

                # settings['flux_normalized_termination_cutoff'] = 0.05
                # settings['flux_normalized_termination_cutoff'] = 0.01
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
                settings['Rm'] = 10.0

                settings['use_RF_terms'] = True
                settings['RF_rc'] = RF_rc_curr
                settings['RF_lc'] = RF_lc_curr
                settings['RF_cr'] = RF_cr_curr
                settings['RF_cl'] = RF_cl_curr
                settings['RF_rl'] = RF_rl_curr
                settings['RF_lr'] = RF_lr_curr

                settings['save_dir'] = sub_folder + '/' + run_name
                print('save dir: ' + str(settings['save_dir']))
                os.makedirs(settings['save_dir'], exist_ok=True)
                os.chdir(settings['save_dir'])

                command = rate_eqs_script + ' --settings "' + str(settings) + '"'
                s = Slurm(run_name, slurm_kwargs=slurm_kwargs)
                s.run(command)
                cnt += 1
                print('run # ' + str(cnt) + ' / ' + str(total_number_of_combinations))

                os.chdir(pwd)
