import pickle
import time

import numpy as np
from scipy.io import savemat, loadmat

from plot_functions import plot_relaxation_status, plot_relaxation_end
from rate_functions import calculate_transition_density, get_density_time_derivatives, \
    get_isentrope_temperature, get_thermal_velocity, \
    get_coulomb_scattering_rate, get_mirror_cell_sizes, \
    get_transmission_rate, calculate_mean_free_path, \
    get_mmm_velocity, get_mmm_drag_rate, define_loss_cone_fractions, \
    get_fluxes


def find_rate_equations_steady_state(settings):
    """ find the rate equations steady state using a relaxation algorithm """
    start_time = time.time()

    # initialize densities
    settings['n_transition'] = calculate_transition_density(settings['n0'], settings['Ti_0'], settings['Te_0'],
                                                            settings)
    settings['n_end'] = 1.0 * settings['n_transition']
    state = initialize_densities(settings)

    # initialize temperatures
    state['Ti'] = get_isentrope_temperature(state['n'], settings, species='ions')
    state['Te'] = get_isentrope_temperature(state['n'], settings, species='electrons')

    # relaxation algorithm initialization
    t_curr = 0
    num_steps = 0
    status_counter = 0
    state['termination_criterion_reached'] = False

    print('*** Begin relaxation iterations ***')
    while t_curr < settings['t_stop']:

        state['v_th'] = get_thermal_velocity(state['Ti'], settings, species='ions')
        state['coulomb_scattering_rate'] = get_coulomb_scattering_rate(state['n'], state['Ti'], state['Te'], settings,
                                                                       species='ions')
        state['mean_free_path'] = calculate_mean_free_path(state['n'], state['Ti'], state['Te'], settings, state=state,
                                                           species='ions')
        state['mirror_cell_sizes'] = get_mirror_cell_sizes(state['n'], state['Ti'], state['Te'], settings, state=state)
        state['transmission_rate'] = get_transmission_rate(state['v_th'], state['mirror_cell_sizes'])
        state['U'] = get_mmm_velocity(state, settings)
        state['mmm_drag_rate'] = get_mmm_drag_rate(state, settings)
        state['alpha_tL'], state['alpha_tR'], state['alpha_c'] = define_loss_cone_fractions(state, settings)
        state['dn_c_dt'], state['dn_tL_dt'], state['dn_tR_dt'] = get_density_time_derivatives(state, settings)

        # advance step
        dt = define_time_step(state, settings)
        t_curr += dt
        num_steps += 1
        state = advance_densities_time_step(state, settings, dt)

        # boundary conditions
        state = enforce_boundary_conditions(state, settings)

        # update temperatures
        state['Ti'] = get_isentrope_temperature(state['n'], settings, species='ions')
        state['Te'] = get_isentrope_temperature(state['n'], settings, species='electrons')

        if check_status_threshold_passed(state, settings, t_curr, num_steps, status_counter):
            # print basic information
            if settings['print_time_step_info'] is True:
                print_time_step_info(dt, t_curr, num_steps)

            # define fluxes and check if termination criterion is reached
            state = get_fluxes(state, settings)
            if check_termination_criterion_reached(state, settings, status_counter):
                state['termination_criterion_reached'] = True

            # plot status
            if settings['do_plot_status'] is True:
                plot_relaxation_status(state, settings)
                status_counter += 1

        if state['termination_criterion_reached'] is True:
            break

    print('*** Finished relaxation iterations ***')
    if state['termination_criterion_reached'] is not True:
        print('\x1b[5;30;41m' + 'Termination criterion was NOT reached.' + '\x1b[0m')
    else:
        print('\x1b[5;30;42m' + 'Termination criterion was reached.' + '\x1b[0m')

    # finalize the generated plots with plot settings
    if settings['do_plot_status'] is True:
        plot_relaxation_end()

    # run time
    state['run_time'] = get_simulation_time(start_time)
    state['t_end'] = t_curr
    state['num_time_steps'] = num_steps
    print('Relaxation t_end = ' + str(state['t_end']))
    print('Relaxation num_time_steps = ' + str(state['num_time_steps']))

    # save results
    if settings['save_state'] is True:
        save_simulation(state, settings)

    return state


def initialize_densities(settings):
    state = {}
    alpha = 1 / settings['Rm']
    r = (1 - alpha) / alpha
    state['n_c'] = np.linspace(settings['n0'] * r / (1 + r), settings['n_end'] * r / (1 + r), settings['N'])
    state['n_tL'] = np.linspace(settings['n0'] / (1 + r) / 2, settings['n_end'] / (1 + r) / 2, settings['N'])
    state['n_tR'] = np.linspace(settings['n0'] / (1 + r) / 2, settings['n_end'] / (1 + r) / 2, settings['N'])
    state['n'] = state['n_c'] + state['n_tL'] + state['n_tR']
    return state


def define_time_step(state, settings):
    dt_c = np.abs(state['n_c'] / state['dn_c_dt'])
    dt_tL = np.abs(state['n_tL'] / state['dn_tL_dt'])
    dt_tR = np.abs(state['n_tR'] / state['dn_tR_dt'])
    dt = settings['dt_factor'] * min(min(dt_c), min(dt_tL), min(dt_tR))

    if dt <= 0:
        raise ValueError('invalid negative dt=' + str(dt) + '. Sign of a problem.')

    if dt <= settings['dt_min']:
        raise ValueError('dt=' + str(dt) + '. Small time step is a sign of a problem or some inefficiency.')

    return dt


def print_time_step_info(dt, t_curr, num_time_steps):
    print('************')
    print('dt = ' + str(dt))
    print('t_curr = ' + str(t_curr))
    print('num_time_steps = ' + str(num_time_steps))
    return


def advance_densities_time_step(state, settings, dt):
    for var_name in ['n_c', 'n_tL', 'n_tR']:
        der_var_name = 'd' + var_name + '_dt'
        state[var_name] = state[var_name] + state[der_var_name] * dt

        if min(state[var_name]) < settings['n_min']:
            raise ValueError('min(' + var_name + ') = ' + str(min(state[var_name])) + '. Sign of a problem.')

    return state


def enforce_boundary_conditions(state, settings):
    # left boundary condition
    if settings['left_boundary_condition'] == 'enforce_tR':
        state['n_tR'][0] = settings['n0'] - state['n_c'][0] - state['n_tL'][0]
    elif settings['left_boundary_condition'] == 'uniform_scaling':
        state['n_c'][0] = state['n_c'][0] * settings['n0'] / state['n'][0]
        state['n_tL'][0] = state['n_tL'][0] * settings['n0'] / state['n'][0]
        state['n_tR'][0] = state['n_tR'][0] * settings['n0'] / state['n'][0]
    else:
        raise TypeError('invalid left_boundary_condition = ' + settings['left_boundary_condition'])

    # right boundary condition
    if settings['right_boundary_condition'] == 'enforce_tL':
        state['n_tL'][-1] = settings['n_end'] - state['n_c'][-1] - state['n_tR'][-1]
    elif settings['right_boundary_condition'] == 'uniform_scaling':
        state['n_c'][-1] = state['n_c'][-1] * settings['n_end'] / state['n'][-1]
        state['n_tL'][-1] = state['n_tL'][-1] * settings['n_end'] / state['n'][-1]
        state['n_tR'][-1] = state['n_tR'][-1] * settings['n_end'] / state['n'][-1]
    else:
        raise TypeError('invalid right_boundary_condition = ' + settings['right_boundary_condition'])

    # update total densities
    state['n'] = state['n_c'] + state['n_tL'] + state['n_tR']
    return state


def check_status_threshold_passed(state, settings, t_curr, num_time_steps, plot_counter):
    if num_time_steps == 1 or t_curr >= plot_counter * settings['dt_print']:
        return True
    else:
        return False


def check_termination_criterion_reached(state, settings, status_counter):
    if state['flux_normalized_std'] < settings['flux_normalized_termination_cutoff'] and status_counter >= 1:
        return True
    else:
        return False


def get_simulation_time(start_time):
    end_time = time.time()
    run_time = end_time - start_time
    print('run_time = ' + str(run_time) + ' sec, ' + str(run_time / 60.0) + ' min.')
    return run_time


def save_simulation(state, settings):
    print('Saving the state and settings of the simulation.')
    if settings['save_format'] == 'pickle':
        with open(settings['save_dir'] + '/' + settings['state_file'] + '.pickle', 'wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(settings['save_dir'] + '/' + settings['settings_file'] + '.pickle', 'wb') as handle:
            pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif settings['save_format'] == 'mat':
        savemat(settings['save_dir'] + '/' + settings['state_file'] + '.mat', state)
        savemat(settings['save_dir'] + '/' + settings['settings_file'] + '.mat', settings)
    else:
        raise TypeError('invalid save_format = ' + settings['save_format'])
    return


def load_simulation(state_file, settings_file, save_format='pickle'):
    if save_format == 'pickle':
        with open(state_file, 'rb') as fid:
            state = pickle.load(fid)
        with open(settings_file, 'rb') as fid:
            settings = pickle.load(fid)
    elif save_format == 'mat':
        state = loadmat(state_file)
        settings = loadmat(settings_file)
    else:
        raise TypeError('invalid save_format = ' + save_format)
    return state, settings
