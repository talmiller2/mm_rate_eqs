import matplotlib.pyplot as plt
import os
from default_settings import define_default_settings
from relaxation_algorithm_functions import find_rate_equations_steady_state

# parametric scan
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.5_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.01_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1_DT_mix/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_mfp_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'
save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_mfp_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'

if not os.path.exists(save_dir_main):
    os.mkdir(save_dir_main)


# for Rm in [1.4, 2.0, 3.0]:
# for Rm in [2.0]:
# for Rm in [2.5]:
for Rm in [3.0]:
# for Rm in [3.5, 4.0, 4.5, 5.0]:
    # for U0 in [0, 1e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5]:
    # for U0 in [7e5, 8e5, 9e5, 1e6]:
    # for U0 in [0, 1e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6]:
    # for U0 in [1e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6]:
    for U0 in [0]:
    # for U0 in [1e4, 1e5]:
    # for U0 in [2e5, 3e5, 4e5, 5e5, 6e5]:
    # for U0 in [6e5, 7e5]:
    # for U0 in [8e5, 9e5, 1e6]:

        print('Rm=' + str(Rm) + ', U0=' + str(U0))

        settings = {'Rm': Rm, 'U0': U0}
        settings['save_dir'] = save_dir_main + '/Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0'])

        if U0 == 0:
            settings['number_of_cells'] = 300
        elif U0 < 1e5:
            settings['number_of_cells'] = 100
        else:
            settings['number_of_cells'] = 50

        settings['transition_density_factor'] = 0.1
        # settings['transition_density_factor'] = 0.5
        # settings['transition_density_factor'] = 0.01

        # settings['delta_n_smoothing_factor'] = 0.01
        # settings['delta_n_smoothing_factor'] = 0.05
        settings['delta_n_smoothing_factor'] = 0.1

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

        plt.close('all')

        settings = define_default_settings(settings)
        # state = find_rate_equations_steady_state(settings)

