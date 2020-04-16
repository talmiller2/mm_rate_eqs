import matplotlib.pyplot as plt
import os
from default_settings import define_default_settings
from relaxation_algorithm_functions import find_rate_equations_steady_state
import numpy as np

# parametric scan
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.5_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.01_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1_DT_mix/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_mfp_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'
save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3/'

if not os.path.exists(save_dir_main):
    os.mkdir(save_dir_main)


# Rm_list = np.array([2.0])
# Rm_list = np.array([2.5])
Rm_list = np.array([3.0])
# Rm_list = np.array([2.0, 2.5])
# Rm_list = np.array([2.0, 2.5, 3.0])
# Rm_list = np.array([2.5, 3.0])
# U0_list = np.array([0])
# U0_list = np.array([0.5])
U0_list = np.array([0.7])
# U0_list = np.array([0.8])
# U0_list = np.array([0.1])
# U0_list = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
# U0_list = np.array([0, 0.05, 0.1, 0.2])

for Rm in Rm_list:
    for U0 in U0_list:

        print('Rm=' + str(Rm) + ', U0=' + str(U0))

        settings = {'Rm': Rm, 'U0': U0}
        # settings['save_dir'] = save_dir_main + '/Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0'])
        settings['save_dir'] = save_dir_main + '/' + 'Rm_' + str(settings['Rm']) + '_U_rel_' + str(settings['U0'])

        if U0 == 0:
            settings['number_of_cells'] = 300
        elif U0 < 1e5:
            settings['number_of_cells'] = 100
        else:
            settings['number_of_cells'] = 50

        settings['transition_density_factor'] = 0.1
        # settings['transition_density_factor'] = 0.5
        # settings['transition_density_factor'] = 0.01

        settings['delta_n_smoothing_factor'] = 0.01
        # settings['delta_n_smoothing_factor'] = 0.05
        # settings['delta_n_smoothing_factor'] = 0.1

        settings['gas_name'] = 'hydrogen'
        # settings['gas_name'] = 'DT_mix'

        settings['n0'] = 1e22  # m^-3
        settings['Ti_0'] = 3 * 1e3 # eV
        settings['Te_0'] = 1 * 1e3 # eV
        settings['n_min'] = 1e15
        settings['ion_velocity_factor'] = 1.0
        settings['cell_size'] = 3.0  # m (MMM wavelength)

        settings['adaptive_mirror'] = 'adjust_cell_size_with_vth'
        # settings['adaptive_mirror'] = 'adjust_cell_size_with_mfp'

        if settings['adaptive_mirror'] == 'adjust_cell_size_with_mfp':
            settings['number_of_cells'] = 2 * settings['number_of_cells']

        # settings['plasma_dimension'] = 1.0
        settings['plasma_dimension'] = 3.0

        plt.close('all')

        settings = define_default_settings(settings)
        state = find_rate_equations_steady_state(settings)

