import numpy as np


def define_default_settings(settings={}):

    # physical constants
    settings['lnCoulombLambda'] = 10.0
    settings['eV'] = 1.0
    settings['keV'] = 1e3 * settings['eV']
    settings['eV_to_K'] = 1.16e4

    # plasma parameters
    if 'plasma_gas' not in settings:
        settings['plasma_gas'] = 'hydrogen'
    settings['me'] = 9.10938356e-31  # kg

    if settings['plasma_gas'] == 'hydrogen':
        settings['mi'] = 1836.9796 * settings['me']
        settings['A_atomic_weight'] = 1.00784
    elif settings['plasma_gas'] == 'deuterium':
        settings['mi'] = 3674.7844 * settings['me']
        settings['A_atomic_weight'] = 2.01410177811
    elif settings['plasma_gas'] == 'tritium':
        settings['mi'] = 5502.6724 * settings['me']
        settings['A_atomic_weight'] = 3.0160492
    elif settings['plasma_gas'] == 'DT_mix':
        settings['mi'] = np.mean([3674.7844, 5502.6724]) * settings['me'] #some approximation
        settings['A_atomic_weight'] = np.mean([2.01410177811, 3.0160492])
    elif settings['plasma_gas'] == 'lithium':
        settings['mi'] = 12651.2892 * settings['me']
        settings['A_atomic_weight'] = 6.941
    elif settings['plasma_gas'] == 'potassium':
        settings['mi'] = 71822.7794 * settings['me']
        settings['A_atomic_weight'] = 39.0983
    else:
        raise TypeError('invalid gas: ' +  settings['plasma_gas'])

    # system parameters
    settings['n0'] = 1e22  #m^-3
    settings['Ti_0'] = 3 * settings['keV']
    settings['Te_0'] = 1 * settings['keV']
    settings['transition_density_factor'] = 0.1
    settings['delta_n_smoothing'] = 0.1
    settings['cell_size'] = 3.0 # m (MMM wavelength)
    settings['N'] = 100

    # options
    # settings['uniform_system'] = True
    settings['uniform_system'] = False

    settings['transition_type'] = 'none'
    # settings['transition_type'] = 'smooth_transition_to_uniform'
    # settings['transition_type'] = 'smooth_transition_to_tR'
    # settings['transition_type'] = 'sharp_transition_to_tR'

    # settings['adaptive_mirror'] = 'adjust_lambda'
    # settings['adaptive_mirror'] = 'adjust_U'
    settings['adaptive_mirror'] = 'None'

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

    return settings