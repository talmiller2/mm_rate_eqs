import matplotlib.pyplot as plt
import os
from default_settings import define_default_settings
from relaxation_algorithm_functions import find_rate_equations_steady_state
import numpy as np

# parametric scan

# save_dir_main = 'runs/runs_April_2020/runs_no_transition_different_number_of_cells/'
# number_of_cells_list = np.round(np.linspace(5,150,15))

# nullify_ntL_factor = 0.05
# nullify_ntL_factor = 0.01
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_nullify_ntL_factor_' \
#                 + str(nullify_ntL_factor)
save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none'
number_of_cells_list = np.round(np.linspace(5, 100, 15))

os.makedirs(save_dir_main, exist_ok=True)

for number_of_cells in number_of_cells_list:

    print('number_of_cells=' + str(int(number_of_cells)))

    settings = {}
    settings['assume_constant_temperature'] = True
    settings['number_of_cells'] = int(number_of_cells)
    # settings['right_boundary_condition'] = 'nullify_ntL'
    # settings['nullify_ntL_factor'] = nullify_ntL_factor
    settings['right_boundary_condition'] = 'none'
    settings['energy_conservation_scheme'] = 'none'
    settings['save_dir'] = save_dir_main + '/number_of_cells_' + str(int(number_of_cells))
    plt.close('all')

    if not os.path.isdir(settings['save_dir']):
        settings = define_default_settings(settings)
        state = find_rate_equations_steady_state(settings)
