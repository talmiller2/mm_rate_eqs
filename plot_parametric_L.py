import matplotlib.pyplot as plt
import numpy as np

from default_settings import define_default_settings
from relaxation_algorithm_functions import load_simulation

plt.close('all')

# parametric scan
save_dir_main = 'runs/runs_no_transition_different_number_of_cells/'


number_of_cells_list = np.array([10, 30, 50])
flux_list = np.zeros(len(number_of_cells_list))

for i, number_of_cells in enumerate(number_of_cells_list):
    print('number_of_cells=' + str(number_of_cells))

    save_dir = save_dir_main + '/number_of_cells_' + str(number_of_cells)

    state_file = save_dir + '/state.pickle'
    settings_file = save_dir + '/settings.pickle'
    state, settings = load_simulation(state_file, settings_file)

    flux_list[i] = state['flux_mean']

plt.figure(1)
plt.plot(number_of_cells_list, flux_list, '-o')

plt.figure(1)
plt.xlabel('number of cells')
plt.ylabel('flux')
plt.tight_layout()
plt.grid(True)
plt.legend()
