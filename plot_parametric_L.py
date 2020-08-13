import matplotlib.pyplot as plt
import numpy as np

from default_settings import define_default_settings
from relaxation_algorithm_functions import load_simulation

# plt.close('all')

# parametric scan
# save_dir_main = 'runs/runs_April_2020/runs_no_transition_different_number_of_cells/'
# number_of_cells_list = np.round(np.linspace(5,150,15))[0:-1]

# nullify_ntL_factor = 0.05
nullify_ntL_factor = 0.01
label = 'isothermal_nullify_ntL_factor_' + str(nullify_ntL_factor)
save_dir_main = 'runs/runs_August_2020/different_number_of_cells_nullify_ntL_factor_' \
                + str(nullify_ntL_factor)
# label = 'isothermal_rbc_none_energycons_none'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none'
number_of_cells_list = np.round(np.linspace(5, 100, 15))

flux_list = np.zeros(len(number_of_cells_list))

for i, number_of_cells in enumerate(number_of_cells_list):
    # print('number_of_cells=' + str(int(number_of_cells)))

    save_dir = save_dir_main + '/number_of_cells_' + str(int(number_of_cells))

    state_file = save_dir + '/state.pickle'
    settings_file = save_dir + '/settings.pickle'
    state, settings = load_simulation(state_file, settings_file)

    flux_list[i] = state['flux_mean']

plt.figure(1)
plt.plot(number_of_cells_list, flux_list, '-o', label=label)

# p = np.polyfit(number_of_cells_list, flux_list, deg=2)
# flux_poly2_fit = np.polyval(p, number_of_cells_list)
# plt.plot(number_of_cells_list, flux_poly2_fit, '-', label='quad fit')

plt.figure(1)
plt.xlabel('number of cells')
plt.ylabel('flux')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.grid(True)
