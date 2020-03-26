import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle

from default_settings import define_default_settings
from relaxation_algorithm_functions import find_rate_equations_steady_state, plot_relaxation_end, load_simulation

### test the algorithm
settings = define_default_settings()
plt.close('all')
state = find_rate_equations_steady_state(settings)
# plot_relaxation_end()

### test loading
# save_format = 'mat'
# save_format = 'pickle'
save_format = 'None'
state_file = settings['save_dir'] + '/' + settings['state_file'] + '.' + save_format
settings_file = settings['save_dir'] + '/' + settings['settings_file'] + '.' + save_format
state1, settings1 = load_simulation(state_file, settings_file, save_format=save_format)

# state2 = find_rate_equations_steady_state(settings1)
