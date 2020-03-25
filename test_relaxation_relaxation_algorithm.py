import matplotlib.pyplot as plt

from default_settings import define_default_settings
from relaxation_algorithm_functions import find_rate_equations_steady_state, plot_relaxation_end

### test the algorithm
settings = define_default_settings()
plt.close('all')
state = find_rate_equations_steady_state(settings)
# plot_relaxation_end()
