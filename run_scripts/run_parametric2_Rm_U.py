import os

import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.relaxation_algorithm_functions import find_rate_equations_steady_state

# parametric scan
# save_dir_main = 'runs/small_experiment_0.3eV_adaptive_mirror/'
save_dir_main = 'runs/small_experiment_0.3eV/'
# save_dir_main = 'runs/small_experiment_0.3eV_adaptive_mirror_cell_size_10cm/'
# save_dir_main = 'runs/small_experiment_2eV_adaptive_mirror/'
# save_dir_main = 'runs/small_experiment_2eV/'

if not os.path.exists(save_dir_main):
    os.mkdir(save_dir_main)

# Rm_list = np.array([2.0])
Rm_list = np.array([2.5])
# Rm_list = np.array([3.0])
# Rm_list = np.array([2.0, 2.5])
# Rm_list = np.array([2.0, 2.5, 3.0])
# U0_list = np.array([0])
# U0_list = np.array([0.5])
# U0_list = np.array([0.05])
# U0_list = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
U0_list = np.array([0, 0.05, 0.1, 0.2])

for Rm in Rm_list:
    for U0 in U0_list:

        print('Rm=' + str(Rm) + ', U0=' + str(U0))

        settings = {'Rm': Rm, 'U0': U0}

        if U0 < 0.1:
            settings['number_of_cells'] = 200
            settings['dt_status'] = 1e-2
        else:
            settings['number_of_cells'] = 50
            settings['dt_status'] = 1e-2
        settings['t_stop'] = 1e-1

        settings['transition_density_factor'] = 0.1
        # settings['transition_density_factor'] = 0.5
        # settings['transition_density_factor'] = 0.01

        # settings['delta_n_smoothing_factor'] = 0.01
        settings['delta_n_smoothing_factor'] = 0.05
        # settings['delta_n_smoothing_factor'] = 0.1

        # settings['gas_name'] = 'hydrogen'
        # settings['gas_name'] = 'helium'
        settings['gas_name'] = 'lithium'

        settings['n0'] = 1e16  # m^-3
        settings['Ti_0'] = 0.3  # eV
        settings['Te_0'] = 0.3  # eV
        # settings['n0'] = 5e17  # m^-3
        # settings['Ti_0'] = 2.0 # eV
        # settings['Te_0'] = 2.0 # eV
        settings['n_min'] = settings['n0'] / 1000
        settings['ion_velocity_factor'] = np.sqrt(2)
        settings['cell_size'] = 0.05  # m (MMM wavelength)
        # settings['cell_size'] = 0.1  # m (MMM wavelength)

        settings['adaptive_mirror'] = 'none'
        # settings['adaptive_mirror'] = 'adjust_cell_size_with_vth'

        settings['save_dir'] = save_dir_main + '/' + settings['gas_name'] \
                               + '_Rm_' + str(settings['Rm']) + '_U_rel_' + str(settings['U0'])

        plt.close('all')

        settings = define_default_settings(settings)
        state = find_rate_equations_steady_state(settings)
