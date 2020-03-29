import logging

import numpy as np

from loss_cone_functions import get_solid_angles


def theta_fun(x):
    if x > 0:
        return x
    else:
        return 0


def get_gamma_dimension(d=1):
    """
    gamma of ideal gas EOS, in d spatial dimensions
    From Bellan 'Fundamentals of Plasma Physics' p. 50
    """
    return (d + 2.0) / d


def get_transition_filters(n, settings):
    f1 = (1 + np.exp(-(n - settings['n_transition']) / settings['delta_n_smoothing']))
    f2 = (1 + np.exp((n - settings['n_transition']) / settings['delta_n_smoothing']))
    return f1, f2


def get_isentrope_temperature(n, settings, species='ions'):
    """
    calculate the temperature based on the density, assuming the plasma expands
    adiabatically on the isentrope of ideal gas EOS.
    """
    n0 = settings['n0']
    if species == 'ions':
        T0 = settings['Ti_0']
    else:
        T0 = settings['Te_0']

    if settings['uniform_system'] is True:
        return T0 + 0 * n
    else:
        if settings['adaptive_dimension'] is True:
            n_trans = settings['n_transition']
            gamma_start = get_gamma_dimension(1)
            gamma_fin = get_gamma_dimension(3)
            T_trans = T0 * (n_trans / n0) ** (gamma_start - 1)
            f1, f2 = get_transition_filters(n, settings)
            return T0 * (n / n0) ** (gamma_start - 1) / f1 \
                   + T_trans * (n / n_trans) ** (gamma_fin - 1) / f2
        else:
            return T0 * (n / n0) ** (get_gamma_dimension(1) - 1)


def get_dominant_temperature(Ti, Te, settings, quick_mode=True):
    if quick_mode is True:
        # if the temperature difference is not too large, the electrons have the dominant thermal velocity
        return Te
    else:
        if type(Te) is not type(Ti):
            raise TypeError('Both temperature arguments need to be of consistent type.')
        if type(Te) is float or type(Te) is np.float64 or type(Te) is int:
            if Ti / Te > settings['mi'] / settings['me']:
                return Ti
            else:
                return Te
        elif type(Te) is np.ndarray:
            Tie = np.zeros(len(Te))
            for i in range(len(Te)):
                if Ti[i] / Te[i] > settings['mi'] / settings['me']:
                    Tie[i] = Ti[i]
                else:
                    Tie[i] = Ti[i]
            return Tie
        else:
            raise TypeError('invalid type(Te) = ' + str(type(Te)))
        return Tie


def get_thermal_velocity(T, settings, species='ions'):
    if species == 'ions':
        species_factor = 1.0 * settings['ion_velocity_factor']
        T0 = settings['Ti_0']
    else:
        species_factor = np.sqrt(settings['mi'] / settings['me']) * settings['electron_velocity_factor']
        T0 = settings['Te_0']

    if settings['uniform_system'] is True:
        return 1.4e4 * species_factor * np.sqrt(T0 / settings['A_atomic_weight']) + 0 * T
    else:
        return 1.4e4 * species_factor * np.sqrt(T / settings['A_atomic_weight'])


def get_coulomb_scattering_rate(n, Ti, Te, settings, species='ions'):
    """
    Coulomb scattering rate (Bellan 'Fundamentals of Plasma Physics' p. 15, 21)
    """
    Tie = get_dominant_temperature(Ti, Te, settings)

    if settings['uniform_system'] is True:
        if type(n) is not float:
            n = n[0] + 0 * n

    Zi = settings['Z_ion']
    Ze = 1.0
    if species == 'ions':
        i_on_i_factor = settings['ion_scattering_rate_factor'] * Zi ** 2 / np.sqrt(settings['mi'] / settings['me'])
        i_on_e_factor = settings['ion_scattering_rate_factor'] * Zi * Ze / (settings['mi'] / settings['me'])
        return 4e-12 * settings['lnCoulombLambda'] * n * (Ti ** (-1.5) * i_on_i_factor + Tie ** (-1.5) * i_on_e_factor)
    else:
        e_on_i_factor = settings['electron_scattering_rate_factor'] * Zi * Ze
        e_on_e_factor = settings['electron_scattering_rate_factor'] * Ze ** 2
        return 4e-12 * settings['lnCoulombLambda'] * n * (Tie ** (-1.5) * e_on_i_factor + Te ** (-1.5) * e_on_e_factor)


def calculate_mean_free_path(n, Ti, Te, settings, state=None, species='ions'):
    if state is None:
        v_th = get_thermal_velocity(Ti, settings, species=species)
        nu_s = get_coulomb_scattering_rate(n, Ti, Te, settings, species=species)
        return v_th / nu_s
    elif 'v_th' in state and 'coulomb_scattering_rate' in state:
        return state['v_th'] / state['coulomb_scattering_rate']
    else:
        raise TypeError('invalid option.')


def get_mirror_cell_sizes(n, Ti, Te, settings, state=None):
    if settings['adaptive_mirror'] == 'adjust_lambda':
        if state is not None and 'mean_free_path' in state:
            mfp = state['mean_free_path']
        else:
            mfp = calculate_mean_free_path(n, Ti, Te, settings, state=state)
        mirror_cell_sizes = settings['cell_size_mfp_factor'] * mfp
    else:
        mirror_cell_sizes = settings['cell_size'] + 0 * Ti
    return mirror_cell_sizes


def get_transmission_rate(v_th, mirror_cell_sizes):
    return v_th / mirror_cell_sizes


def calculate_transition_density(n, Ti, Te, settings, state=None):
    if state is not None and 'mean_free_path' in state:
        mfp = state['mean_free_path']
    else:
        mfp = calculate_mean_free_path(n, Ti, Te, settings, state=state)
    return settings['n0'] * (settings['transition_density_factor'] * settings['cell_size'] / mfp) ** (
            1 / (2 * get_gamma_dimension(1) - 3))


def get_mmm_velocity(state, settings):
    if settings['adaptive_mirror'] == 'adjust_U':
        if 'v_th' not in state:
            U = settings['U0'] * get_thermal_velocity(state['Ti'], settings) / get_thermal_velocity(settings['Ti_0'],
                                                                                                    settings)
        else:
            U = settings['U0'] * state['v_th'] / state['v_th'][0]
    else:
        U = settings['U0'] + 0 * state['v_th']
    return U


def get_mmm_drag_rate(state, settings):
    n = state['n']
    U = state['U']
    n_trans = settings['n_transition']
    delta_n_smoothing = settings['delta_n_smoothing']

    if settings['transition_type'] == 'none':
        return U / state['mirror_cell_sizes']
    elif settings['transition_type'] in ['smooth_transition_to_uniform', 'smooth_transition_to_tR']:
        f1 = (1 + np.exp(-(n - n_trans) / delta_n_smoothing))
        return U / state['mirror_cell_sizes'] / f1
    elif settings['transition_type'] == 'sharp_transition_to_tR':
        U_mod = U
        for i in range(len(n)):
            if n[i] < n_trans:
                U_mod[i] = 0
        return U_mod / state['mirror_cell_sizes']
    else:
        raise ValueError('invalid transition_type: ' + settings['transition_type'])


def define_loss_cone_fractions(state, settings):
    v_th = state['v_th']
    U = state['U']
    alpha = 1 / settings['Rm']
    if settings['alpha_definition'] == 'old_constant':
        alpha_tL = alpha / 2.0 + 0 * v_th
        alpha_tR = alpha / 2.0 + 0 * v_th
        alpha_c = 1.0 - alpha_tL - alpha_tR
    elif settings['alpha_definition'] == 'geometric_constant':
        alpha_tR, alpha_tL, alpha_c = get_solid_angles(settings['U0'], v_th[0], alpha)
        alpha_tR = alpha_tR + 0 * v_th
        alpha_tL = alpha_tL + 0 * v_th
        alpha_c = alpha_c + 0 * v_th
    elif settings['alpha_definition'] == 'geometric_local':
        alpha_tL = 0 * v_th
        alpha_tR = 0 * v_th
        alpha_c = 0 * v_th
        for i in range(len(v_th)):
            alpha_tR[i], alpha_tL[i], alpha_c[i] = get_solid_angles(U[i], v_th[i], alpha)
    else:
        raise ValueError('invalid alpha_definition: ' + settings['alpha_definition'])
    return alpha_tL, alpha_tR, alpha_c


def get_density_time_derivatives(state, settings):
    """
    The core of the relaxation algorithm, define the rate equations that advances the state variables
    """

    # state variables
    n = state['n']
    n_c = state['n_c']
    n_tL = state['n_tL']
    n_tR = state['n_tR']
    n_trans = settings['n_transition']
    nu_s = state['coulomb_scattering_rate']
    nu_t = state['transmission_rate']
    nu_d = state['mmm_drag_rate']
    alpha_tL = state['alpha_tL']
    alpha_tR = state['alpha_tR']
    alpha_c = state['alpha_c']

    # initializations
    f_scat_c = np.zeros(settings['N'])
    f_scat_tL = np.zeros(settings['N'])
    f_scat_tR = np.zeros(settings['N'])
    f_trans_L = np.zeros(settings['N'])
    f_trans_R = np.zeros(settings['N'])
    f_drag = np.zeros(settings['N'])

    # transition filters
    f1, f2 = get_transition_filters(n, settings)

    # define density time derivative
    if settings['transition_type'] == 'smooth_transition_to_uniform':
        # smooth transition from the normal rates to a uniform description of the three sub-species,
        # while turning off the MMM drag and L,R transmission to a right-only transport
        f_trans_uniform = np.zeros(settings['N'])
        for k in range(settings['N']):
            f_scat_c[k] = + nu_s[k] * (alpha_c[k] * (n_tL[k] + n_tR[k])
                                       - (alpha_tL[k] + alpha_tR[k]) * n_c[k]) / f1[k] \
                          + nu_s[k] / 3 * (n_tL[k] + n_tR[k] - 2 * n_c[k]) / f2[k]

            f_scat_tL[k] = + nu_s[k] * (-(alpha_c[k] + alpha_tR[k]) * n_tL[k]
                                        + alpha_tL[k] * n_tR[k]
                                        + alpha_tL[k] * n_c[k]) / f1[k] \
                           + nu_s[k] / 3 * (- 2 * n_tL[k] + n_tR[k] + n_c[k]) / f2[k]

            f_scat_tR[k] = + nu_s[k] * (-(alpha_c[k] + alpha_tL[k]) * n_tR[k]
                                        + alpha_tR[k] * n_tL[k]
                                        + alpha_tR[k] * n_c[k]) / f1[k] \
                           + nu_s[k] / 3 * (n_tL[k] - 2 * n_tR[k] + n_c[k]) / f2[k]

            f_trans_L[k] = - nu_t[k] * n_tL[k] / f1[k]
            if k < settings['N'] - 1:
                f_trans_L[k] = f_trans_L[k] + nu_t[k + 1] * n_tL[k + 1] / f1[k + 1]

            f_trans_R[k] = - nu_t[k] * n_tR[k] / f1[k]
            if k > 0:
                f_trans_R[k] = f_trans_R[k] + nu_t[k - 1] * n_tR[k - 1] / f1[k - 1]

            f_trans_uniform[k] = + 1.0 / 3 * (- nu_t[k] * n[k]) / f2[k]
            if k > 0:
                f_trans_uniform[k] = f_trans_uniform[k] + 1.0 / 3 * (nu_t[k - 1] * n[k - 1]) / f2[k - 1]

            f_drag[k] = - nu_d[k] * n_c[k]
            if k < settings['N'] - 1:
                f_drag[k] = f_drag[k] + nu_d[k + 1] * n_c[k + 1]
            else:
                f_drag[k] = f_drag[k] + f_drag[k - 1]

        # combine rates
        dn_c_dt = f_scat_c + f_drag + f_trans_uniform
        dn_tL_dt = f_scat_tL + f_trans_L + f_trans_uniform
        dn_tR_dt = f_scat_tR + f_trans_R + f_trans_uniform

    elif settings['transition_type'] in ['none', 'smooth_transition_to_tR', 'sharp_transition_to_tR']:
        # keeping the same rates, but changing (or not) the left transmission term
        for k in range(settings['N']):
            f_scat_c[k] = + nu_s[k] * (alpha_c[k] * (n_tL[k] + n_tR[k])
                                       - (alpha_tL[k] + alpha_tR[k]) * n_c[k])

            f_scat_tL[k] = + nu_s[k] * (-(alpha_c[k] + alpha_tR[k]) * n_tL[k]
                                        + alpha_tL[k] * n_tR[k]
                                        + alpha_tL[k] * n_c[k])

            f_scat_tR[k] = + nu_s[k] * (-(alpha_c[k] + alpha_tL[k]) * n_tR[k]
                                        + alpha_tR[k] * n_tL[k]
                                        + alpha_tR[k] * n_c[k])

            if settings['transition_type'] == 'none':
                f_trans_L[k] = - nu_t[k] * n_tL[k]
                if k < settings['N'] - 1:
                    f_trans_L[k] = f_trans_L[k] + nu_t[k + 1] * n_tL[k + 1]
            elif settings['transition_type'] == 'smooth_transition_to_tR':
                f_trans_L[k] = - nu_t[k] * n_tL[k] / f1[k]
                if k < settings['N'] - 1:
                    f_trans_L[k] = f_trans_L[k] + nu_t[k + 1] * n_tL[k + 1] / f1[k + 1]
            elif settings['transition_type'] == 'sharp_transition_to_tR':
                if n[k] > n_trans:  # shut off left flux below threshold
                    f_trans_L[k] = - nu_t[k] * n_tL[k]
                    if k < settings['N'] - 1:
                        f_trans_L[k] = f_trans_L[k] + nu_t[k + 1] * n_tL[k + 1]

            f_trans_R[k] = - nu_t[k] * n_tR[k]
            if k > 0:
                f_trans_R[k] = f_trans_R[k] + nu_t[k - 1] * n_tR[k - 1]

            f_drag[k] = - nu_d[k] * n_c[k]
            if k < settings['N'] - 1:
                f_drag[k] = f_drag[k] + nu_d[k + 1] * n_c[k + 1]
            else:
                f_drag[k] = f_drag[k] + f_drag[k - 1]

        # combine rates
        dn_c_dt = f_scat_c + f_drag
        dn_tL_dt = f_scat_tL + f_trans_L
        dn_tR_dt = f_scat_tR + f_trans_R
    else:
        raise TypeError('invalid transition_type = ' + settings['transition_type'])

    # dni_c_dt = dni_c_dt + 1e-50
    # dni_tL_dt = dni_tL_dt + 1e-50
    # dni_tR_dt = dni_tR_dt + 1e-50

    return dn_c_dt, dn_tL_dt, dn_tR_dt


def get_fluxes(state, settings):
    # state variables
    n = state['n']
    n_c = state['n_c']
    n_tL = state['n_tL']
    n_tR = state['n_tR']
    v_th = state['v_th']
    U = state['U']
    n_trans = settings['n_transition']

    # initializations
    flux_trans_R = np.nan * np.zeros(settings['N'])
    flux_trans_L = np.nan * np.zeros(settings['N'])
    flux_mmm_drag = np.nan * np.zeros(settings['N'])

    # transition filters
    f1, f2 = get_transition_filters(n, settings)

    # calculate fluxes (multiplied by 2 because there are 2 plugs to the main cell)
    flux_factor = 2.0 * settings['cross_section_main_cell']

    if settings['transition_type'] == 'smooth_transition_to_uniform':
        flux_trans_uniform = np.nan * np.zeros(settings['N'])
        for k in range(0, settings['N'] - 1):
            flux_trans_R[k] = v_th[k] * n_tR[k] / f1[k] * flux_factor
            flux_trans_L[k] = - v_th[k + 1] * n_tL[k + 1] / f1[k + 1] * flux_factor
            flux_trans_uniform[k] = (v_th[k] * n[k] / f2[k]) * flux_factor
            flux_mmm_drag[k] = (- U[k + 1] * n_c[k + 1]) * flux_factor
        flux = flux_trans_R + flux_trans_L + flux_trans_uniform + flux_mmm_drag

    elif settings['transition_type'] in ['none', 'smooth_transition_to_tR', 'sharp_transition_to_tR']:
        for k in range(0, settings['N'] - 1):
            flux_trans_R[k] = v_th[k] * n_tR[k] / f1[k] * flux_factor
            flux_mmm_drag[k] = (- U[k + 1] * n_c[k + 1]) * flux_factor

            if settings['transition_type'] == 'none':
                flux_trans_L[k] = - v_th[k + 1] * n_tL[k + 1] * flux_factor
            elif settings['transition_type'] == 'smooth_transition_to_tR':
                flux_trans_L[k] = - v_th[k + 1] * n_tL[k + 1] / f1[k + 1] * flux_factor
            elif settings['transition_type'] == 'sharp_transition_to_tR':
                if n[k] > n_trans:  # shut off left flux below threshold
                    flux_trans_L[k] = - v_th[k + 1] * n_tL[k + 1] * flux_factor
                else:
                    flux_trans_L[k] = 0
        flux = flux_trans_R + flux_trans_L + flux_mmm_drag
    else:
        raise TypeError('invalid transition_type = ' + settings['transition_type'])

    # save fluxes to state
    state['flux_trans_R'] = flux_trans_R
    state['flux_trans_L'] = flux_trans_L
    state['flux_mmm_drag'] = flux_mmm_drag
    state['flux'] = flux

    # calculate flux  statistics
    state['flux_max'] = np.nanmax(flux)
    state['flux_min'] = np.nanmin(flux)
    state['flux_mean'] = np.nanmean(flux)
    state['flux_std'] = np.nanstd(flux)
    state['flux_normalized_std'] = state['flux_std'] / state['flux_mean']

    # print('Flux statistics:')
    # logging.info('flux_mean = ' + '{:.2e}'.format(state['flux_mean']))
    # logging.info('flux_normalized_std = ' + '{:.2e}'.format(state['flux_normalized_std']))
    logging.info('flux_mean = ' + '{:.2e}'.format(state['flux_mean']) + ', flux_normalized_std = ' + '{:.2e}'.format(
        state['flux_normalized_std']))

    return state
