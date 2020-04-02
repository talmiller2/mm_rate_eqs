import matplotlib.pyplot as plt

from default_settings import define_default_settings
from relaxation_algorithm_functions import find_rate_equations_steady_state
from relaxation_algorithm_functions import load_simulation, plot_relaxation_status, plot_relaxation_end

### test the algorithm
# settings = {'gas_name': 'hydrogen'}
# settings = define_default_settings(settings)
settings = define_default_settings()
plt.close('all')

state = find_rate_equations_steady_state(settings)


### test loading
# save_format = 'mat'
# save_format = 'pickle'
# save_format = 'None'
# state_file = settings['save_dir'] + '/' + settings['state_file'] + '.' + save_format
# settings_file = settings['save_dir'] + '/' + settings['settings_file'] + '.' + save_format
# state1, settings1 = load_simulation(state_file, settings_file, save_format=save_format)


# state2 = find_rate_equations_steady_state(settings1)

#plot an additional run next to current one
# save_dir = 'runs/test_Rm_3.0_U_3.0e+05_smooth_transition_adjust_cell_size_right_bc_uniform_scaling'
# save_dir = 'runs/test_Rm_3.0_U_3.0e+05_smooth_transition_adjust_cell_size'
# state_file = save_dir + '/state.pickle'
# settings_file = save_dir + '/settings.pickle'
# state_load, settings_load = load_simulation(state_file, settings_file)
# plot_relaxation_status(state_load, settings_load)
# plot_relaxation_end(settings_load)