import numpy as np


def define_default_settings(settings=None):
    if settings == None:
        settings = {}

    #### physical constants
    settings['lnCoulombLambda'] = 10.0
    settings['eV'] = 1.0
    settings['keV'] = 1e3 * settings['eV']
    settings['eV_to_K'] = 1.16e4
    settings['MeV_to_J'] = 1e6 * 1.6e-19
    settings['kB_K'] = 1.380649e-23 #J/K
    settings['kB_eV'] = settings['kB_K'] * settings['eV_to_K']  # J/eV

    ### plasma parameters
    if 'plasma_gas' not in settings:
        settings['plasma_gas'] = 'hydrogen'
    settings['me'], settings['mi'], settings['A_atomic_weight'], settings['Z_charge'] \
        = define_plasma_parameters(gas_name=settings['plasma_gas'])

    ### system parameters
    if 'n0' not in settings:
        settings['n0'] = 1e22  # m^-3
    if 'Ti_0' not in settings:
        settings['Ti_0'] = 3 * settings['keV']
    if 'Te_0' not in settings:
        settings['Te_0'] = 1 * settings['keV']
    settings['B'] = 1.0 * np.sqrt( settings['n0'] / 1e20 * settings['Ti_0'] / (5 * settings['keV']) )  # [Tesla]
    settings['transition_density_factor'] = 0.1
    settings['delta_n_smoothing'] = 0.1
    settings['cell_size'] = 3.0  # m (MMM wavelength)
    settings['N'] = 100
    settings['length_main_cell'] = 100 # m
    settings['diameter_main_cell'] = 0.5 # m
    settings['cross_section_main_cell'] = np.pi*(settings['diameter_main_cell']/2)**2 # m
    settings['volume_main_cell'] = settings['length_main_cell'] * settings['cross_section_main_cell'] # m^3

    ### additional options

    # settings['uniform_system'] = True
    settings['uniform_system'] = False

    settings['adaptive_dimension'] = False
    # settings['adaptive_dimension'] = True

    settings['transition_type'] = 'none'
    # settings['transition_type'] = 'smooth_transition_to_uniform'
    # settings['transition_type'] = 'smooth_transition_to_tR'
    # settings['transition_type'] = 'sharp_transition_to_tR'

    settings['adaptive_mirror'] = 'None'
    # settings['adaptive_mirror'] = 'adjust_lambda'
    # settings['adaptive_mirror'] = 'adjust_U'

    # settings['alpha_definition'] = 'old_constant'
    settings['alpha_definition'] = 'geometric_constant'
    # settings['alpha_definition'] = 'geometric_local'

    settings['cell_size_mfp_factor'] = 1.0
    settings['ion_velocity_factor'] = 1.0
    # settings['ion_velocity_factor'] = np.sqrt(2)
    settings['electron_velocity_factor'] = 1.0
    settings['ion_scattering_rate_factor'] = 1.0
    settings['electron_scattering_rate_factor'] = 1.0
    settings['cell_size_mfp_factor'] = 1.0

    ### relaxation solver parameters
    settings['t_stop'] = 1e-1
    settings['t_solve_min'] = 1e-20
    settings['dt_print'] = 1e-5
    settings['dt_factor'] = 0.3
    settings['dt_min'] = 1e-20
    settings['n_min'] = 1e10

    settings['left_boundary_condition'] = 'enforce_tR'
    # settings['left_boundary_condition'] = 'uniform_scaling'
    settings['right_boundary_condition'] = 'enforce_tL'
    # settings['right_boundary_condition'] = 'uniform_scaling'

    settings['flux_normalized_termination_cutoff'] = 0.05

    settings['do_plot_status'] = True
    settings['save_state'] = False
    settings['state_save_file'] = 'runs/state.pickle'

    return settings


def define_plasma_parameters(gas_name='hydrogen'):
    me = 9.10938356e-31  # kg
    mp = 1.67262192e-27  # kg
    if gas_name == 'hydrogen':
        A = 1.00784
        Z = 1.0
    elif gas_name == 'deuterium':
        A = 2.01410177811
        Z = 1.0
    elif gas_name == 'tritium':
        A = 3.0160492
        Z = 1.0
    elif gas_name == 'DT_mix':
        A = np.mean([2.01410177811, 3.0160492])  # some approximation
        Z = 1.0
    elif gas_name == 'lithium':
        A = 6.941
        Z = 3.0
    elif gas_name == 'potassium':
        A = 39.0983
        Z = 19.0
    else:
        raise TypeError('invalid gas: ' + gas_name)
    mi = A * mp
    return me, mi, A, Z
