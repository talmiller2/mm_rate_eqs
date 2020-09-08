import logging
import os
import pickle
import time

import numpy as np
from scipy.io import savemat, loadmat

from mm_rate_eqs.fusion_functions import get_ideal_gas_pressure, \
    get_magnetic_field_for_given_pressure
from mm_rate_eqs.plot_functions import plot_relaxation_status, save_plots
from mm_rate_eqs.rate_functions import calculate_transition_density, \
    get_density_time_derivatives, \
    get_isentrope_temperature, \
    get_thermal_velocity, \
    get_coulomb_scattering_rate, \
    get_mirror_cell_sizes, \
    get_transmission_velocities, \
    calculate_mean_free_path, \
    get_mmm_velocity, \
    define_loss_cone_fractions, \
    get_fluxes, \
    get_transition_filters, \
    get_collective_velocity


def find_rate_equations_steady_state(settings):
    """ find the rate equations steady state using a relaxation algorithm """
    start_time = time.time()
    initialize_logging(settings)

    # initialize densities
    settings, state = initialize_densities(settings)

    # initialize temperatures
    state['Ti'] = get_isentrope_temperature(state['n'], settings, species='ions')
    state['Te'] = get_isentrope_temperature(state['n'], settings, species='electrons')

    # relaxation algorithm initialization
    t_curr = 0
    num_time_steps = 0
    status_counter = 0
    state['termination_criterion_reached'] = False
    state['successful_termination'] = False

    while t_curr < settings['t_stop']:

        state['v_th'] = get_thermal_velocity(state['Ti'], settings, species='ions')
        state['coulomb_scattering_rate'] = get_coulomb_scattering_rate(state['n'], state['Ti'], state['Te'], settings,
                                                                       species='ions')
        state['mean_free_path'] = calculate_mean_free_path(state['n'], state['Ti'], state['Te'], settings, state=state,
                                                           species='ions')
        state['mirror_cell_sizes'] = get_mirror_cell_sizes(state['n'], state['Ti'], state['Te'], settings, state=state)
        state['v_col'], state['flux_E'] = get_collective_velocity(state, settings)
        state['f_above'], state['f_below'] = get_transition_filters(state['n'], settings)  # transition filters
        state['v_R'], state['v_L'] = get_transmission_velocities(state, settings)
        state['U'] = get_mmm_velocity(state, settings)
        state['alpha_tR'], state['alpha_tL'], state['alpha_c'] = define_loss_cone_fractions(state, settings)
        state['dn_c_dt'], state['dn_tL_dt'], state['dn_tR_dt'] = get_density_time_derivatives(state, settings)

        # variables for plots
        state['transmission_rate_R'] = state['v_R'] / state['mirror_cell_sizes']
        state['transmission_rate_L'] = state['v_L'] / state['mirror_cell_sizes']
        state['mmm_drag_rate'] = state['U'] / state['mirror_cell_sizes']

        # print basic run info
        if num_time_steps == 0:
            print_basic_run_info(state, settings)
            logging.info('***  Begin relaxation iterations  ***')

        # advance step
        dt, state = define_time_step(state, settings)
        t_curr += dt
        num_time_steps += 1
        state = advance_densities_time_step(state, settings, dt, t_curr, num_time_steps)

        # boundary conditions
        state = enforce_boundary_conditions(state, settings)

        # update temperatures
        state['Ti'] = get_isentrope_temperature(state['n'], settings, species='ions')
        state['Te'] = get_isentrope_temperature(state['n'], settings, species='electrons')

        if check_status_threshold_passed(settings, t_curr, num_time_steps, status_counter) \
                or state['termination_criterion_reached']:
            # print basic information
            if settings['print_time_step_info'] is True:
                print_time_step_info(dt, t_curr, num_time_steps)

            # print minimal density values for debugging
            logging.info('min: n=' + '{:.2e}'.format(min(state['n']))
                         + ', n_c=' + '{:.2e}'.format(min(state['n_c']))
                         + ', n_tL=' + '{:.2e}'.format(min(state['n_tL']))
                         + ', n_tR=' + '{:.2e}'.format(min(state['n_tR'])))

            # define fluxes and check if termination criterion is reached
            state = get_fluxes(state, settings)
            state = save_fluxes_evolution(state, t_curr)
            state = check_termination_criterion_reached(state, settings, t_curr, num_time_steps, status_counter)

            # plot status
            if settings['draw_plots'] is True:
                plot_relaxation_status(state, settings)
                if settings['save_plots_scheme'] == 'status_plots':
                    save_plots(settings)

            status_counter += 1

        if state['termination_criterion_reached'] is True:
            break

    logging.info('*************************************')
    logging.info('*** Finished relaxation iterations ***')
    if state['successful_termination'] == False:
        # print in red color
        logging.info('\x1b[5;30;41m' + 'Termination unsuccessful.' + '\x1b[0m')
    else:
        # print in green color
        logging.info('\x1b[5;30;42m' + 'Termination successful.' + '\x1b[0m')

    # save the plots
    if settings['draw_plots'] is True:
        if settings['save_plots_scheme'] == 'status_plots':
            save_plots(settings)
        elif settings['save_plots_scheme'] == 'only_at_calculation_end':
            plot_relaxation_status(state, settings)
            save_plots(settings)
        else:
            raise ValueError('draw_plots is True but save_plots_scheme is invalid. '
                             'save_plots_scheme = ' + str(settings['save_plots_scheme']))

    # run time
    state['run_time'] = get_simulation_time(start_time)
    state['t_end'] = t_curr
    state['num_time_steps'] = num_time_steps

    # save results
    if settings['save_state'] is True:
        save_simulation(state, settings)

    return state


def initialize_logging(settings):
    if 'log_file' in settings:
        # create save directory and log file
        if not os.path.exists(settings['save_dir']):
            os.mkdir(settings['save_dir'])
        log_file_path = settings['save_dir'] + '/' + settings['log_file'] + '.txt'

        # basic logging definition
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # remove any previously defined loggers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # print log messages to a log file
        fh = logging.FileHandler(log_file_path)
        fh_formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%d-%m-%y %H:%M:%S')
        fh.setFormatter(fh_formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        # print log messages to the console (screen)
        ch = logging.StreamHandler()
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        # disable the debug logs generated by matplotlib
        logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

    else:
        # print log messages to the console only
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    return


def initialize_densities(settings):
    settings['n_transition'] = calculate_transition_density(settings['n0'], settings['Ti_0'], settings['Te_0'],
                                                            settings)
    settings['theoretical_n_transition'] = settings['n_transition']

    # definitions for right boundary density
    if settings['right_boundary_condition_density_type'] == 'none':
        settings['n_end'] = settings['n0'] / 5.0  # just for the initial guess
    elif settings['right_boundary_condition_density_type'] == 'n_transition':
        settings['n_end'] = settings['n_transition']
        if settings['n_transition'] >= settings['n0']:
            raise ValueError('n_transition < n0, meaning leftmost cell has too short mfp / cell size.')
        if settings['n_transition'] < settings['n_end_min']:
            logging.info('n_transition=' + str(settings['n_transition']) + ' < n_end_min=' + str(settings['n_end_min'])
                         + ', changing n_end and n_transition to n_end_min.')
            settings['n_end'] = settings['n_end_min']
            settings['n_transition'] = settings['n_end_min']

    elif settings['right_boundary_condition_density_type'] == 'n_expander':
        settings['n_end'] = settings['n0'] / 20.0  # approximating a low value
    else:
        raise TypeError(
            'invalid right_boundary_condition_density_type = ' + settings['right_boundary_condition_density_type'])

    # initial density profiles
    state = {}
    if settings['initialization_type'] == 'linear_uniform':
        for var_name in ['n_c', 'n_tL', 'n_tR']:
            state[var_name] = np.linspace(settings['n0'], settings['n_end'], settings['number_of_cells'])
            state[var_name] /= 3.0
    elif settings['initialization_type'] == 'linear_alpha':
        alpha = 1 / settings['Rm']
        r = (1 - alpha) / alpha
        state['n_c'] = np.linspace(settings['n0'] * r / (1 + r), settings['n_end'] * r / (1 + r),
                                   settings['number_of_cells'])
        state['n_tL'] = np.linspace(settings['n0'] / (1 + r) / 2, settings['n_end'] / (1 + r) / 2,
                                    settings['number_of_cells'])
        state['n_tR'] = np.linspace(settings['n0'] / (1 + r) / 2, settings['n_end'] / (1 + r) / 2,
                                    settings['number_of_cells'])
    elif settings['initialization_type'] == 'FD_decay':
        N_array = np.linspace(1, settings['number_of_cells'], settings['number_of_cells'])
        for var_name in ['n_c', 'n_tL', 'n_tR']:
            state[var_name] = settings['n0'] + (settings['n_end'] - settings['n0']) / (
                    1 + np.exp(-(N_array - 0.4 * settings['number_of_cells']) / (0.1 * settings['number_of_cells'])))
            state[var_name] /= 3.0
    else:
        raise TypeError('invalid initialization_type = ' + settings['initialization_type'])
    state['n'] = state['n_c'] + state['n_tL'] + state['n_tR']
    return settings, state


def print_basic_run_info(state, settings):
    logging.info('*************************************')
    logging.info('***********  Plasma info  ***********')
    logging.info('*************************************')
    logging.info('n0 = ' + str('{:.2e}'.format(settings['n0'])) + ' m^-3')
    logging.info('Ti_0 = ' + str(settings['Ti_0']) + ' eV')
    logging.info('Te_0 = ' + str(settings['Te_0']) + ' eV')
    settings['P'] = get_ideal_gas_pressure(settings['n0'], settings['Ti_0'], settings)
    logging.info('P = ' + str(settings['P']) + ' bar')
    settings['B'] = get_magnetic_field_for_given_pressure(settings['P'], beta=1.0)  # [Tesla]
    logging.info('B = ' + str(settings['B']) + ' T (for beta=1)')
    logging.info('v_th = ' + str('{:.2e}'.format(state['v_th'][0])) + ' m/s')
    logging.info('mfp = ' + str('{:.2e}'.format(state['mean_free_path'][0])) + ' m')
    logging.info('cell_size = ' + str('{:.2e}'.format(state['mirror_cell_sizes'][0])) + ' m')
    logging.info('mfp/cell_size = ' + str(state['mean_free_path'][0] / state['mirror_cell_sizes'][0]))

    logging.info('*************************************')
    logging.info('***********  System info  ***********')
    logging.info('*************************************')
    logging.info('number_of_cells = ' + str(settings['number_of_cells']))
    logging.info('Rm = ' + str(settings['Rm']))
    logging.info('U0 = ' + str(settings['U0']))

    logging.info('*************************************')
    logging.info('****  Physics model assumptions  ****')
    logging.info('*************************************')
    logging.info('assume_constant_density = ' + str(settings['assume_constant_density']))
    logging.info('assume_constant_temperature = ' + str(settings['assume_constant_temperature']))
    logging.info('assume_constant_transmission = ' + str(settings['assume_constant_transmission']))
    logging.info('plasma_dimension = ' + str(settings['plasma_dimension']))
    logging.info('transition_type = ' + str(settings['transition_type']))
    logging.info('transmission_factor = ' + str(settings['transmission_factor']))
    logging.info('alpha_definition = ' + str(settings['alpha_definition']))
    logging.info('adaptive_mirror = ' + str(settings['adaptive_mirror']))

    logging.info('*************************************')
    return


def define_time_step(state, settings):
    dt_list = []
    for var_name in ['n_c', 'n_tL', 'n_tR']:
        der_var_name = 'd' + var_name + '_dt'
        # prevent the derivatives from being exactly zero, so in the division by it  an error will not happen
        state[der_var_name] += 1e-50
        dt_list += [min(np.abs(state[var_name] / state[der_var_name]))]
    dt = settings['dt_factor'] * min(dt_list)

    if dt <= 0:
        raise ValueError('invalid negative dt=' + str(dt) + '. Sign of a problem.')

    if dt <= settings['dt_min']:
        # raise ValueError('dt=' + str(dt) + '. Small time step is a sign of a problem or some inefficiency.')
        print('dt=' + str(dt) + '. Small time step is a sign of a problem or some inefficiency.')
        state['termination_criterion_reached'] = True
        state['successful_termination'] = False

    return dt, state


def print_time_step_info(dt, t_curr, num_time_steps):
    logging.info('*************************************')
    logging.info('dt = ' + str(dt) + ', t_curr = ' + str(t_curr) + ', num_time_steps = ' + str(num_time_steps))
    return


def advance_densities_time_step(state, settings, dt, t_curr, num_time_steps):
    for var_name in ['n_c', 'n_tL', 'n_tR']:
        der_var_name = 'd' + var_name + '_dt'
        state[var_name] = state[var_name] + state[der_var_name] * dt

        if settings['fail_on_minimal_density'] is True:
            if min(state[var_name]) < settings['n_min']:
                print_time_step_info(dt, t_curr, num_time_steps)
                raise ValueError('min(' + var_name + ') = ' + str(min(state[var_name])) + '. Sign of a problem.')
        else:
            ind_min = np.where(state[var_name] < settings['n_min'])
            state[var_name][ind_min] = settings['n_min']

    state['n'] = state['n_c'] + state['n_tL'] + state['n_tR']

    return state


def enforce_boundary_conditions(state, settings):
    # left boundary condition
    if settings['left_boundary_condition'] == 'adjust_ntR_for_n0':
        state['n_tR'][0] = settings['n0'] - state['n_c'][0] - state['n_tL'][0]
    elif settings['left_boundary_condition'] == 'adjust_all_species_for_n0':
        state['n_c'][0] = state['n_c'][0] * settings['n0'] / state['n'][0]
        state['n_tL'][0] = state['n_tL'][0] * settings['n0'] / state['n'][0]
        state['n_tR'][0] = state['n_tR'][0] * settings['n0'] / state['n'][0]
    elif settings['left_boundary_condition'] == 'none':
        pass
    else:
        raise TypeError('invalid left_boundary_condition = ' + settings['left_boundary_condition'])

    # right boundary condition
    if settings['right_boundary_condition'] == 'adjust_ntL_for_nend':
        state['n_tL'][-1] = settings['n_end'] - state['n_c'][-1] - state['n_tR'][-1]
    elif settings['right_boundary_condition'] == 'adjust_all_species_for_nend':
        state['n_c'][-1] = state['n_c'][-1] * settings['n_end'] / state['n'][-1]
        state['n_tL'][-1] = state['n_tL'][-1] * settings['n_end'] / state['n'][-1]
        state['n_tR'][-1] = state['n_tR'][-1] * settings['n_end'] / state['n'][-1]
    elif settings['right_boundary_condition'] == 'nullify_ntL':
        # assign a low value to n_tL to enforce low left flux, but keep the other densities free
        state['n_tL'][-1] = state['n_tL'][0] * settings['nullify_ntL_factor']
    elif settings['right_boundary_condition'] == 'none':
        pass
    else:
        raise TypeError('invalid right_boundary_condition = ' + settings['right_boundary_condition'])

    # update total densities
    state['n'] = state['n_c'] + state['n_tL'] + state['n_tR']
    return state


def check_status_threshold_passed(settings, t_curr, num_time_steps, status_counter):
    if num_time_steps == 1:
        return True
    elif settings['status_counter_type'] == 'time_steps':
        if num_time_steps >= status_counter * settings['time_steps_status']:
            return True
    elif settings['status_counter_type'] == 'time_elapsed':
        if t_curr >= status_counter * settings['dt_status']:
            return True
    else:
        return False


def save_fluxes_evolution(state, t_curr):
    vars_list = ['t', 'flux_max', 'flux_min', 'flux_mean', 'flux_std', 'flux_normalized_std']
    for var_name in vars_list:
        var_name_evolution = var_name + '_evolution'

        # initialization
        if var_name_evolution not in state:
            state[var_name_evolution] = []

        # append progress
        if var_name is 't':
            state[var_name_evolution] += [t_curr]
        else:
            state[var_name_evolution] += [state[var_name]]

    return state


def check_termination_criterion_reached(state, settings, t_curr, num_time_steps, status_counter):
    if state['flux_normalized_std'] < settings['flux_normalized_termination_cutoff'] \
            and status_counter >= 1 and t_curr >= settings['t_solve_min']:
        logging.info('flux_normalized_termination_cutoff reached.')
        state['termination_criterion_reached'] = True
        state['successful_termination'] = True
    elif num_time_steps > settings['max_num_time_steps']:
        logging.info('max_num_time_steps reached.')
        state['termination_criterion_reached'] = True
        state['successful_termination'] = False
    else:
        pass
    return state


def get_simulation_time(start_time):
    end_time = time.time()
    run_time = end_time - start_time
    logging.info('run_time = ' + str(run_time) + ' sec, ' + str(run_time / 60.0) + ' min.')
    return run_time


def save_simulation(state, settings):
    # create save directory
    if not os.path.exists(settings['save_dir']):
        logging.info('Creating save directory: ' + settings['save_dir'])
        os.mkdir(settings['save_dir'])

    # save file
    logging.info('Saving the state and settings of the simulation (' + settings['save_format'] + ' format).')
    if settings['save_format'] == 'pickle':
        with open(settings['save_dir'] + '/' + settings['state_file'] + '.pickle', 'wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(settings['save_dir'] + '/' + settings['settings_file'] + '.pickle', 'wb') as handle:
            pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif settings['save_format'] == 'mat':
        # note: the mat format puts everything into matrix format [[]], which means we cannot load the settings
        # and immediately rerun them, some post-processing in necessary.
        savemat(settings['save_dir'] + '/' + settings['state_file'] + '.mat', state)
        savemat(settings['save_dir'] + '/' + settings['settings_file'] + '.mat', settings)
    else:
        raise TypeError('invalid save format = ' + settings['save_format'])
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
        raise TypeError('invalid save format = ' + save_format)
    return state, settings
