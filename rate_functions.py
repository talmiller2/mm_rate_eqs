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
    """
    "Fermi-Dirac" type smoothing function
    """
    delta_n_smoothing = settings['delta_n_smoothing_factor'] * settings['n0']
    f1 = (1 + np.exp(-(n - settings['n_transition']) / delta_n_smoothing))
    f2 = (1 + np.exp((n - settings['n_transition']) / delta_n_smoothing))
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
            return T0 * (n / n0) ** (get_gamma_dimension(settings['plasma_dimension']) - 1)


def get_dominant_temperature(Ti, Te, settings, quick_mode=True):
    """
    define the relevant temperature to use in the expression for the ion-electron Coulomb scattering rate
    """
    if quick_mode is True:
        # if the temperature difference is not too large, the electrons have the larger thermal velociry,
        # and therefore define the dominant temperature
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
    if species == 'ions':
        i_on_i_factor = settings['ion_scattering_rate_factor'] * Zi ** 4 / np.sqrt(settings['mi'] / settings['me'])
        i_on_e_factor = settings['ion_scattering_rate_factor'] * Zi ** 2 / (settings['mi'] / settings['me'])
        return 4e-12 * settings['lnCoulombLambda'] * n * (Ti ** (-1.5) * i_on_i_factor + Tie ** (-1.5) * i_on_e_factor)
    else:
        e_on_i_factor = settings['electron_scattering_rate_factor'] * Zi ** 2
        e_on_e_factor = settings['electron_scattering_rate_factor']
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
    if settings['adaptive_mirror'] == 'adjust_cell_size_with_mfp':
        if state is not None and 'mean_free_path' in state:
            mfp = state['mean_free_path']
        else:
            mfp = calculate_mean_free_path(n, Ti, Te, settings, state=state)
        return settings['cell_size'] * mfp / mfp[0]

    elif settings['adaptive_mirror'] == 'adjust_cell_size_with_vth':
        if state is not None and 'v_th' in state:
            v_th = state['v_th']
        else:
            v_th = get_thermal_velocity(Ti, settings)
        return settings['cell_size'] * v_th / v_th[0]

    else:
        return settings['cell_size'] + 0 * Ti


def get_transmission_rate(v_th, mirror_cell_sizes):
    return v_th / mirror_cell_sizes


def calculate_transition_density(n, Ti, Te, settings, state=None):
    if state is not None and 'mean_free_path' in state:
        mfp = state['mean_free_path']
    else:
        mfp = calculate_mean_free_path(n, Ti, Te, settings, state=state)
    return settings['n0'] * (settings['transition_density_factor'] * settings['cell_size'] / mfp) ** (
            1 / (2 * get_gamma_dimension(settings['plasma_dimension']) - 3))


def get_mmm_velocity(state, settings):
    # define MMM velocity for different cells

    if settings['mmm_velocity_type'] == 'absolute':
        U0 = settings['U0']
    elif settings['mmm_velocity_type'] == 'relative_to_thermal_velocity':
        if 'v_th' not in state:
            v_th = get_thermal_velocity(state['Ti'], settings)
        else:
            v_th = state['v_th']
        U0 = settings['U0'] * v_th[0]
    else:
        raise ValueError('invalid mmm_velocity_type: ' + settings['mmm_velocity_type'])

    if settings['adaptive_mirror'] == 'adjust_U':
        # change U at different positions directly. This is only theoretical, because in reality we can define
        # the MMM frequency, and the velocity is derived from the wavelength (U=wavelength*frequency)
        if 'v_th' not in state:
            v_th = get_thermal_velocity(state['Ti'], settings)
        else:
            v_th = state['v_th']
        U = U0 * v_th / v_th[0]

    elif settings['adaptive_mirror'] in ['adjust_cell_size_with_mfp', 'adjust_cell_size_with_vth']:
        # the realistic method, the MMM velocity scales with the cell size (wavelength)
        if 'mirror_cell_sizes' not in state:
            mirror_cell_sizes = get_mirror_cell_sizes(state['n'], state['Ti'], state['Te'], settings, state=state)
        else:
            mirror_cell_sizes = state['mirror_cell_sizes']
        U = U0 * mirror_cell_sizes / mirror_cell_sizes[0]

    else:
        U = U0 + 0 * state['n']

    return U


def get_mmm_drag_rate(state, settings):
    n = state['n']
    U = state['U']
    n_trans = settings['n_transition']

    if settings['transition_type'] == 'none':
        return U / state['mirror_cell_sizes']
    elif settings['transition_type'] in ['smooth_transition_to_uniform', 'smooth_transition_to_tR']:
        f1, f2 = get_transition_filters(n, settings)
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
        alpha_tR, alpha_tL, alpha_c = get_solid_angles(0, v_th[0], alpha)
        alpha_tR = alpha_tR + 0 * v_th
        alpha_tL = alpha_tL + 0 * v_th
        alpha_c = alpha_c + 0 * v_th
    elif settings['alpha_definition'] == 'geometric_constant_U0':
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
    return alpha_tR, alpha_tL, alpha_c


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
    v_th = state['v_th']
    cell_sizes = state['mirror_cell_sizes']
    U = state['mmm_drag_rate'] * state['mirror_cell_sizes']  # effective mirror velocity
    alpha_tL = state['alpha_tL']
    alpha_tR = state['alpha_tR']
    alpha_c = state['alpha_c']

    # initializations
    f_scat_c = np.zeros(settings['number_of_cells'])
    f_scat_tL = np.zeros(settings['number_of_cells'])
    f_scat_tR = np.zeros(settings['number_of_cells'])
    f_trans_L = np.zeros(settings['number_of_cells'])
    f_trans_R = np.zeros(settings['number_of_cells'])
    f_drag = np.zeros(settings['number_of_cells'])

    # transition filters
    f1, f2 = get_transition_filters(n, settings)

    # define density time derivative
    if settings['transition_type'] == 'smooth_transition_to_uniform':
        # smooth transition from the normal rates to a uniform description of the three sub-species,
        # while turning off the MMM drag and L,R transmission to a right-only transport
        f_trans_uniform = np.zeros(settings['number_of_cells'])
        for k in range(settings['number_of_cells']):
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

            f_trans_L[k] = - v_th[k] * n_tL[k] / f1[k] / cell_sizes[k]
            if k < settings['number_of_cells'] - 1:
                f_trans_L[k] = f_trans_L[k] + v_th[k + 1] * n_tL[k + 1] / f1[k + 1] / cell_sizes[k]

            f_trans_R[k] = - v_th[k] * n_tR[k] / f1[k] / cell_sizes[k]
            if k > 0:
                f_trans_R[k] = f_trans_R[k] + v_th[k - 1] * n_tR[k - 1] / f1[k - 1] / cell_sizes[k]

            f_trans_uniform[k] = + 1.0 / 3 * (- v_th[k] * n[k]) / f2[k] / cell_sizes[k]
            if k > 0:
                f_trans_uniform[k] = f_trans_uniform[k] + 1.0 / 3 * (v_th[k - 1] * n[k - 1]) / f2[k - 1] / cell_sizes[k]

            f_drag[k] = - U[k] * n_c[k] / cell_sizes[k]
            if k < settings['number_of_cells'] - 1:
                f_drag[k] = f_drag[k] + U[k + 1] * n_c[k + 1] / cell_sizes[k]
            else:
                f_drag[k] = f_drag[k] + f_drag[k - 1]

        # combine rates
        dn_c_dt = f_scat_c + f_drag + f_trans_uniform
        dn_tL_dt = f_scat_tL + f_trans_L + f_trans_uniform
        dn_tR_dt = f_scat_tR + f_trans_R + f_trans_uniform

    elif settings['transition_type'] in ['none', 'smooth_transition_to_tR', 'sharp_transition_to_tR']:
        # keeping the same rates, but changing (or not) the left transmission term
        for k in range(settings['number_of_cells']):
            f_scat_c[k] = + nu_s[k] * (alpha_c[k] * (n_tL[k] + n_tR[k])
                                       - (alpha_tL[k] + alpha_tR[k]) * n_c[k])

            f_scat_tL[k] = + nu_s[k] * (-(alpha_c[k] + alpha_tR[k]) * n_tL[k]
                                        + alpha_tL[k] * n_tR[k]
                                        + alpha_tL[k] * n_c[k])

            f_scat_tR[k] = + nu_s[k] * (-(alpha_c[k] + alpha_tL[k]) * n_tR[k]
                                        + alpha_tR[k] * n_tL[k]
                                        + alpha_tR[k] * n_c[k])

            if settings['transition_type'] == 'none':
                f_trans_L[k] = - v_th[k] * n_tL[k] / cell_sizes[k]
                if k < settings['number_of_cells'] - 1:
                    f_trans_L[k] = f_trans_L[k] + v_th[k + 1] * n_tL[k + 1] / cell_sizes[k]
            elif settings['transition_type'] == 'smooth_transition_to_tR':
                f_trans_L[k] = - v_th[k] * n_tL[k] / cell_sizes[k] / f1[k]
                if k < settings['number_of_cells'] - 1:
                    f_trans_L[k] = f_trans_L[k] + v_th[k + 1] * n_tL[k + 1] / cell_sizes[k] / f1[k + 1]
            elif settings['transition_type'] == 'sharp_transition_to_tR':
                if n[k] > n_trans:  # shut off left flux below threshold
                    f_trans_L[k] = - v_th[k] * n_tL[k] / cell_sizes[k]
                    if k < settings['number_of_cells'] - 1:
                        f_trans_L[k] = f_trans_L[k] + v_th[k + 1] * n_tL[k + 1] / cell_sizes[k]

            f_trans_R[k] = - v_th[k] * n_tR[k] / cell_sizes[k]
            if k > 0:
                f_trans_R[k] = f_trans_R[k] + v_th[k - 1] * n_tR[k - 1] / cell_sizes[k]

            f_drag[k] = - U[k] * n_c[k] / cell_sizes[k]
            if k < settings['number_of_cells'] - 1:
                f_drag[k] = f_drag[k] + U[k + 1] * n_c[k + 1] / cell_sizes[k]
            else:
                f_drag[k] = f_drag[k] + f_drag[k - 1]

        # f_trans_L = state['flux_trans_L'] / cell_sizes
        # f_trans_R = state['flux_trans_R'] / cell_sizes

        # combine rates
        dn_c_dt = f_scat_c + f_drag
        dn_tL_dt = f_scat_tL + f_trans_L
        dn_tR_dt = f_scat_tR + f_trans_R
    else:
        raise TypeError('invalid transition_type = ' + settings['transition_type'])

    return dn_c_dt, dn_tL_dt, dn_tR_dt


# def get_transmission_fluxes(state, settings):
#     n = state['n']
#     n_tL = state['n_tL']
#     n_tR = state['n_tR']
#     n_trans = settings['n_transition']
#     v_th = state['v_th']
#
#     # transition filters
#     f1, f2 = get_transition_filters(n, settings)
#
#     flux_trans_R = v_th * n_tR
#
#     flux_trans_L = np.zeros(settings['number_of_cells'])
#     for k in range(0, settings['number_of_cells'] - 1):
#         if settings['transition_type'] == 'none':
#             flux_trans_L[k] = - v_th[k + 1] * n_tL[k + 1]
#         elif settings['transition_type'] == 'smooth_transition_to_tR':
#             flux_trans_L[k] = - v_th[k + 1] * n_tL[k + 1] / f1[k + 1]
#         elif settings['transition_type'] == 'sharp_transition_to_tR':
#             if n[k] > n_trans:  # shut off left flux below threshold
#                 flux_trans_L[k] = - v_th[k + 1] * n_tL[k + 1]
#             else:
#                 flux_trans_L[k] = 0
#
#     return flux_trans_R, flux_trans_L


def get_fluxes(state, settings):
    # state variables
    n = state['n']
    n_c = state['n_c']
    n_tL = state['n_tL']
    n_tR = state['n_tR']
    n_trans = settings['n_transition']
    v_th = state['v_th']
    U = state['mmm_drag_rate'] * state['mirror_cell_sizes']  # effective mirror velocity

    # initializations
    flux_trans_R = np.nan * np.zeros(settings['number_of_cells'])
    flux_trans_L = np.nan * np.zeros(settings['number_of_cells'])
    flux_mmm_drag = np.nan * np.zeros(settings['number_of_cells'])

    # transition filters
    f1, f2 = get_transition_filters(n, settings)

    # calculate fluxes (multiplied by 2 because there are 2 plugs to the main cell)
    flux_factor = 2.0 * settings['cross_section_main_cell']

    if settings['transition_type'] == 'smooth_transition_to_uniform':
        flux_trans_uniform = np.nan * np.zeros(settings['number_of_cells'])
        for k in range(0, settings['number_of_cells'] - 1):
            flux_trans_R[k] = v_th[k] * n_tR[k] / f1[k] * flux_factor
            flux_trans_L[k] = - v_th[k + 1] * n_tL[k + 1] / f1[k + 1] * flux_factor
            flux_trans_uniform[k] = (v_th[k] * n[k] / f2[k]) * flux_factor
            flux_mmm_drag[k] = (- U[k + 1] * n_c[k + 1]) * flux_factor
        flux = flux_trans_R + flux_trans_L + flux_trans_uniform + flux_mmm_drag

    elif settings['transition_type'] in ['none', 'smooth_transition_to_tR', 'sharp_transition_to_tR']:
        for k in range(0, settings['number_of_cells'] - 1):
            flux_trans_R[k] = v_th[k] * n_tR[k] * flux_factor
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

        # flux_trans_L = state['flux_trans_L'] * flux_factor
        # flux_trans_R = state['flux_trans_R'] * flux_factor

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
    state['flux_normalized_std'] = np.abs(state['flux_std'] / state['flux_mean'])

    # print('Flux statistics:')
    logging.info('flux_mean = ' + '{:.2e}'.format(state['flux_mean']) + ', flux_normalized_std = ' + '{:.2e}'.format(
        state['flux_normalized_std']))

    return state
