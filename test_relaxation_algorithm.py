import matplotlib.pyplot as plt

from default_settings import define_default_settings
from relaxation_algorithm_functions import find_rate_equations_steady_state
from relaxation_algorithm_functions import load_simulation, plot_relaxation_status, plot_relaxation_end

### test the algorithm
settings = {}
# settings['gas_name'] = 'hydrogen'
# settings['save_state'] = 'False'
# settings['assume_constant_density'] = True
# settings['assume_constant_temperature'] = True
# settings['ion_scattering_rate_factor'] = 10
# settings['cell_size'] = 50
# settings['plasma_dimension'] = 1
# settings['plasma_dimension'] = 1.5
# settings['plasma_dimension'] = 2
settings['plasma_dimension'] = 3
# settings['number_of_cells'] = 20
settings['number_of_cells'] = 30
# settings['number_of_cells'] = 100
# settings['number_of_cells'] = 150
# settings['number_of_cells'] = 200

settings['U0'] = 0
# settings['U0'] = 0.1
# settings['U0'] = 0.3
# settings['U0'] = 0.5
# settings['U0'] = 0.8

# settings['right_boundary_condition'] = 'nullify_ntL'
settings['right_boundary_condition'] = 'none'

settings['nullify_ntL_factor'] = 0.05
# settings['nullify_ntL_factor'] = 0.01

settings['transition_type'] = 'none'
# settings['transition_type'] = 'smooth_transition_to_free_flow'

# settings['energy_conservation_scheme'] = 'none'
# settings['energy_conservation_scheme'] = 'simple'
settings['energy_conservation_scheme'] = 'detailed'

# settings['dt_status'] = 1e-5
settings['dt_status'] = 1e-4
# settings['dt_status'] = 1e-3

settings = define_default_settings(settings)
# settings['n_end_min'] = 0.3 * settings['n0']
plt.close('all')

settings['save_dir'] = 'runs/runs_August_2020/test'
settings['save_dir'] += '_N_' + str(settings['number_of_cells'])
settings['save_dir'] += '_U_' + str(settings['U0'])

if settings['transition_type'] == 'none':
    settings['save_dir'] += '_trans_none'
else:
    settings['save_dir'] += '_trans_smooth'
if settings['assume_constant_temperature'] == True:
    settings['save_dir'] += '_iso'
else:
    settings['save_dir'] += '_cool_d_' + str(settings['plasma_dimension'])

if settings['right_boundary_condition'] == 'nullify_ntL':
    settings['save_dir'] += '_rbc_nullify_ntL'
    settings['save_dir'] += '_factor_' + str(settings['nullify_ntL_factor'])
elif settings['right_boundary_condition'] == 'adjust_ntL_for_nend':
    settings['save_dir'] += '_rbc_adjust_for_nend'
elif settings['right_boundary_condition'] == 'none':
    settings['save_dir'] += '_rbc_none'

settings['save_dir'] += '_energy_scheme_' + settings['energy_conservation_scheme']

state = find_rate_equations_steady_state(settings)

### test loading
# save_format = 'mat'
# save_format = 'pickle'
# save_format = 'None'
# state_file = settings['save_dir'] + '/' + settings['state_file'] + '.' + save_format
# settings_file = settings['save_dir'] + '/' + settings['settings_file'] + '.' + save_format
# state1, settings1 = load_simulation(state_file, settings_file, save_format=save_format)


# state2 = find_rate_equations_steady_state(settings1)

# plot an additional run next to current one
# save_dir = 'runs/runs_August_2020/test_N_100_trans_smooth_iso_U_0.1_rbc_uni'
# state_file = save_dir + '/state.pickle'
# settings_file = save_dir + '/settings.pickle'
# state_load, settings_load = load_simulation(state_file, settings_file)
# plot_relaxation_status(state_load, settings_load)
# plot_relaxation_end(settings_load)