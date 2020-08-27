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
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_const_dens'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_const_dens_n0X0.2'
save_dir_main = '../runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_const_dens_mfpX10'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_U_0.3'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_cool_d_1_U_0'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_cool_d_1_U_0.3'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_cool_d_1_U_0.3_adaptive_mirror_mfp'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_cool_d_3_U_0'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_cool_d_3_U_0.3'
number_of_cells_list = np.round(np.linspace(5, 100, 15))

os.makedirs(save_dir_main, exist_ok=True)

for number_of_cells in number_of_cells_list:

    print('number_of_cells=' + str(int(number_of_cells)))

    settings = {}

    settings['assume_constant_temperature'] = True
    # settings['assume_constant_temperature'] = False

    # settings['assume_constant_density'] = False
    settings['assume_constant_density'] = True

    # settings['n0'] = 3.875e22 / 5 # m^-3 # to increase the mfp in the main cell
    settings['ion_scattering_rate_factor'] = 10  # to increase the mfp in the main cell

    settings['plasma_dimension'] = 1
    # settings['plasma_dimension'] = 3
    settings['number_of_cells'] = int(number_of_cells)
    # settings['right_boundary_condition'] = 'nullify_ntL'
    # settings['nullify_ntL_factor'] = nullify_ntL_factor
    settings['right_boundary_condition'] = 'none'
    settings['energy_conservation_scheme'] = 'none'

    settings['U0'] = 0
    # settings['U0'] = 0.3

    # settings['adaptive_mirror'] = 'adjust_cell_size_with_mfp'
    settings['save_dir'] = save_dir_main + '/number_of_cells_' + str(int(number_of_cells))
    plt.close('all')

    if not os.path.isdir(settings['save_dir']):
        settings = define_default_settings(settings)
        state = find_rate_equations_steady_state(settings)
