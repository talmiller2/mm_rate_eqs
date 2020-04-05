import matplotlib.pyplot as plt
import os
from default_settings import define_default_settings
from relaxation_algorithm_functions import find_rate_equations_steady_state
import numpy as np


# parametric scan
save_dir_main = 'runs/runs_no_transition_different_number_of_cells/'

if not os.path.exists(save_dir_main):
    os.mkdir(save_dir_main)

# number_of_cells_list = np.array([10, 30, 50])
number_of_cells_list = np.round(np.linspace(5,150,15))

for number_of_cells in number_of_cells_list:

    print('number_of_cells=' + str(int(number_of_cells)))

    settings = {'number_of_cells': int(number_of_cells)}
    settings['save_dir'] = save_dir_main + '/number_of_cells_' + str(int(number_of_cells))

    plt.close('all')

    settings = define_default_settings(settings)
    state = find_rate_equations_steady_state(settings)

