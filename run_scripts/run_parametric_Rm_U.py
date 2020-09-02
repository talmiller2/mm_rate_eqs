import matplotlib.pyplot as plt
import os
from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.relaxation_algorithm_functions import find_rate_equations_steady_state
import numpy as np

# parametric scan
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.5_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.01_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1_DT_mix/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_mfp_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_5e20/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_3e21/'

# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_1/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_5e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_3e21/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_15e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_enforce_tL_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_15e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_1e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.001_dim_3_rbc_15e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_enforce_tL_transition_n_factor_0.1_delta_n_factor_0.001_dim_3_rbc_15e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.05_dim_3_rbc_15e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_enforce_tL_transition_n_factor_0.1_delta_n_factor_0.05_dim_3_rbc_15e20/'
save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_2_rbc_15e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_2_rbc_3e21/'
# save_dir_main = 'runs/test3/'

if not os.path.exists(save_dir_main):
    os.mkdir(save_dir_main)

# Rm_list = np.array([2.0])
# Rm_list = np.array([2.5])
Rm_list = np.array([3.0])
# Rm_list = np.array([3.0, 2.5])
# Rm_list = np.array([2.0, 2.5])
# Rm_list = np.array([2.0, 2.5, 3.0])
# Rm_list = np.array([2.5, 3.0])
# U0_list = np.array([0.0])
# U0_list = np.array([0.05])
# U0_list = np.array([0.1])
# U0_list = np.array([0.7])
# U0_list = np.array([0.8])
# U0_list = np.array([0.9])
# U0_list = np.array([1.0])
# U0_list = np.array([1.1])
# U0_list = np.array([0.8, 0.9, 1.0, 1.1])
# U0_list = np.array([0.05, 0.0])
# U0_list = np.array([0.1])
# U0_list = np.array([0.3])
# U0_list = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# U0_list = np.array([0, 0.05, 0.1, 0.3])
U0_list = np.array([0, 0.05, 0.1])
# U0_list = np.array([0.05, 0.1])
# U0_list = np.array([0.05, 0.1, 0.3])
# U0_list = np.array([0.0, 0.05, 0.1, 0.3])
# U0_list = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
# U0_list = np.array([0.1, 0.2, 0.3, 0.4])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4])
# U0_list = np.array([0.2, 0.3, 0.4])
# U0_list = np.array([0.3, 0.4])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
# U0_list = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
# U0_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# U0_list = np.array([0.9, 1.0, 1.1, 0.05, 0])
# U0_list = np.array([1.0, 1.1])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
# U0_list = np.array([0, 0.05, 0.1, 0.2])

for Rm in Rm_list:
    for U0 in U0_list:

        print('Rm=' + str(Rm) + ', U0=' + str(U0))

        settings = {'Rm': Rm, 'U0': U0}
        # settings['save_dir'] = save_dir_main + '/Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0'])
        settings['save_dir'] = save_dir_main + '/' + 'Rm_' + str(settings['Rm']) + '_U_rel_' + str(settings['U0'])

        if U0 < 0.05:
            # settings['number_of_cells'] = 1200
            # settings['number_of_cells'] = 900
            # settings['number_of_cells'] = 700
            # settings['number_of_cells'] = 500
            settings['number_of_cells'] = 300
            # settings['number_of_cells'] = 200
            # settings['number_of_cells'] = 130
            settings['dt_status'] = 1e-2
            # settings['dt_status'] = 1e-3
            settings['dt_factor'] = 0.5
            # settings['dt_factor'] = 0.1
            # settings['t_stop'] = 1.0
            settings['t_stop'] = 100.0
            # settings['initialization_type'] = 'linear_alpha'
        elif U0 < 0.1:
            # settings['number_of_cells'] = 50
            # settings['number_of_cells'] = 300
            settings['number_of_cells'] = 200
            # settings['number_of_cells'] = 100
            settings['dt_status'] = 1e-2
            # settings['dt_status'] = 1e-3
            # settings['dt_status'] = 1e-4
            # settings['dt_status'] = 1e-5
            # settings['dt_factor'] = 0.5
            settings['dt_factor'] = 0.1
            # settings['t_stop'] = 1e-1
            settings['t_stop'] = 10
        else:
            # settings['number_of_cells'] = 50
            settings['number_of_cells'] = 100
            # settings['number_of_cells'] = 200
            # settings['dt_status'] = 1e-2
            settings['dt_status'] = 1e-3
            # settings['dt_status'] = 1e-4
            # settings['dt_status'] = 1e-5
            settings['dt_factor'] = 0.1
            settings['t_stop'] = 1e-1

        # settings['right_boundary_condition'] = 'enforce_tL'
        settings['right_boundary_condition'] = 'uniform_scaling'

        settings['transition_density_factor'] = 0.1
        # settings['transition_density_factor'] = 0.5
        # settings['transition_density_factor'] = 0.01

        # settings['delta_n_smoothing_factor'] = 0.1
        # settings['delta_n_smoothing_factor'] = 0.05
        settings['delta_n_smoothing_factor'] = 0.01
        # settings['delta_n_smoothing_factor'] = 0.001
        # settings['delta_n_smoothing_factor'] = 0.0001

        settings['gas_name'] = 'hydrogen'
        # settings['gas_name'] = 'DT_mix'

        settings['n0'] = 1e22  # m^-3
        settings['Ti_0'] = 3 * 1e3  # eV
        settings['Te_0'] = 1 * 1e3  # eV
        settings['n_min'] = 1e15
        # settings['n_min'] = 1e18
        # settings['n_min'] = 1e19
        # settings['n_min'] = 1e20
        # settings['n_min'] = 1e21
        # settings['n_end_min'] = 1e20
        # settings['n_end_min'] = 5e20
        # settings['n_end_min'] = 3e21
        settings['n_end_min'] = 15e20
        # settings['n_end_min'] = 1e20
        settings['ion_velocity_factor'] = 1.0
        settings['cell_size'] = 3.0  # m (MMM wavelength)

        settings['adaptive_mirror'] = 'none'
        # settings['adaptive_mirror'] = 'adjust_cell_size_with_vth'
        # settings['adaptive_mirror'] = 'adjust_cell_size_with_mfp'

        if settings['adaptive_mirror'] == 'adjust_cell_size_with_mfp':
            settings['number_of_cells'] = 2 * settings['number_of_cells']

        # settings['plasma_dimension'] = 1.0
        settings['plasma_dimension'] = 2.0
        # settings['plasma_dimension'] = 3.0

        plt.close('all')

        settings = define_default_settings(settings)
        state = find_rate_equations_steady_state(settings)
