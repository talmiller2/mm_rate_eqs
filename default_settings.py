import numpy as np


def define_default_settings(settings={}):
    # print(settings)

    # physical constants
    settings['lnCoulombLambda'] = 10.0
    settings['eV'] = 1.0
    settings['keV'] = 1e3 * settings['eV']
    settings['eV_to_K'] = 1.16e4
    settings['MeV_to_J'] = 1e6 * 1.6e-19
    settings['kB_K'] = 1.380649e-23 #J/K
    settings['kB_eV'] = settings['kB_K'] * settings['eV_to_K']  # J/eV

    # plasma parameters
    if 'plasma_gas' not in settings:
        settings['plasma_gas'] = 'hydrogen'

    # print(settings)
    settings['me'], settings['mi'], settings['A_atomic_weight'], settings['Z_charge'] \
        = define_plasma_parameters(gas_name=settings['plasma_gas'])

    # system parameters
    settings['n0'] = 1e22  # m^-3
    settings['Ti_0'] = 3 * settings['keV']
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

    D_main_cell = 100
    d = 0.5
    mirror_cross_section = np.pi * (d / 2) ** 2

    # options
    # settings['uniform_system'] = True
    settings['uniform_system'] = False

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
