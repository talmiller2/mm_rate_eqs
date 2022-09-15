import logging

import numpy as np

from mm_rate_eqs.aux_functions import theta_fun_generic
from mm_rate_eqs.loss_cone_functions import get_solid_angles
from mm_rate_eqs.plasma_functions import define_boltzmann_constant, define_factor_eV_to_K
from mm_rate_eqs.constants_functions import define_electron_charge, define_vacuum_permittivity

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
    f_above = (1 + np.exp(-(n - settings['n_transition']) / delta_n_smoothing))
    f_below = (1 + np.exp((n - settings['n_transition']) / delta_n_smoothing))
    return f_above, f_below


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

    if settings['assume_constant_temperature'] is True:
        return T0 + 0 * n
    else:
        if settings['adaptive_dimension'] is True:
            n_trans = settings['n_transition']
            gamma_start = get_gamma_dimension(1)
            gamma_fin = get_gamma_dimension(3)
            T_trans = T0 * (n_trans / n0) ** (gamma_start - 1)
            f_above, f_below = get_transition_filters(n, settings)
            return T0 * (n / n0) ** (gamma_start - 1) / f_above \
                   + T_trans * (n / n_trans) ** (gamma_fin - 1) / f_below
        else:
            return T0 * (n / n0) ** (get_gamma_dimension(settings['plasma_dimension']) - 1)


def get_thermal_velocity(T, settings, species='ions'):
    '''
    Calculate the thermal velocity of particles according to 1/2*m*v_th^2 = 3/2*kB*T
    '''
    if species == 'ions':
        m = settings['mi']
        if settings['assume_constant_temperature'] is True:
            T = settings['Ti_0'] + 0 * T
    elif species == 'electrons':
        m = settings['me']
        if settings['assume_constant_temperature'] is True:
            T = settings['Te_0'] + 0 * T
    else:
        raise ValueError('invalid option for species = ' + str(species))
    kB_eV = define_boltzmann_constant() * define_factor_eV_to_K()
    return np.sqrt(3.0 * kB_eV * T / m)


def get_collective_velocity(state, settings):
    """
    Plasma in the multi-mirror system requires to conserve energy as it flows along thr axis.
    (No requirement to conserve momentum due to torque applied by the magnetic field).
    Similar to a gas in a tube, as it cools and expands, it gains a collective drift velocity.
    Here we calculate the collective velocity and the energy flux (for plotting).
    """

    v_th = state['v_th']
    Ti = state['Ti']
    n_tR = state['n_tR']
    n_tL = state['n_tL']
    kB_eV = define_boltzmann_constant() * define_factor_eV_to_K()

    mi = settings['mi']  # ion particle mass
    l = state['mirror_cell_sizes']

    if settings['energy_conservation_scheme'] == 'none':
        # no collective velocity
        v_col = 0 * state['v_th']
        flux_E = v_th * (n_tR - n_tL) * kB_eV * Ti

    elif settings['energy_conservation_scheme'] == 'simple':
        # assume the collective velocity is fixed to the main cell thermal velocity
        # in the isothermal approximation, this will be the same as previous case
        v_col = state['v_th'][0] - state['v_th']
        flux_E = v_th * (n_tR - n_tL) * kB_eV * Ti

    elif settings['energy_conservation_scheme'] == 'detailed':
        # when the plasma expands it loses internal energy (even if does not cool),
        # which has to be balanced by extra kinetic energy of the collective velocity.
        # A more detailed energy conservation equation (3rd order polynomical in v_col):
        flux_E = np.nan * np.zeros(settings['number_of_cells'])
        v_col = np.nan * np.zeros(settings['number_of_cells'])
        for k in range(settings['number_of_cells']):
            if k == 0:
                v_col[0] = 0
                flux_E[0] = v_th[0] * (n_tR[0] - n_tL[0]) * (kB_eV * Ti[0] + 0.5 * mi * v_th[0] ** 2.0) / l[0]
            else:
                coef_p_3 = (n_tR[k] + n_tL[k]) * 0.5 * mi / l[k]
                coef_p_2 = (n_tR[k] - n_tL[k]) * 0.5 * mi * v_th[k] / l[k]
                coef_p_1 = (n_tR[k] + n_tL[k]) * kB_eV * Ti[k] / l[k]
                coef_p_0 = (n_tR[k] - n_tL[k]) * kB_eV * Ti[k] * v_th[k] / l[k]
                coef_p_0_shifted = coef_p_0 - flux_E[k - 1]
                p_shifted = [coef_p_3, coef_p_2, coef_p_1, coef_p_0_shifted]
                roots = np.roots(p_shifted)
                indices_real_roots = [i for i in range(len(roots)) if np.isreal(roots[i])]
                if len(indices_real_roots) == 0:
                    raise ValueError('No real roots for energy flux conservation equation,'
                                     'in cell ' + str(k) + '. roots = ' + str(roots))
                elif len(indices_real_roots) > 1:
                    raise ValueError('More than one real root for energy flux conservation equation, '
                                     'in cell ' + str(k) + '. roots = ' + str(roots))
                else:
                    real_root = np.real(roots[indices_real_roots[0]])
                    v_col[k] = np.real(real_root)
                    p = [coef_p_3, coef_p_2, coef_p_1, coef_p_0]
                    flux_E[k] = np.polyval(p, v_col[k])
    else:
        raise TypeError('invalid energy_conservation_scheme = ' + settings['energy_conservation_scheme'])

    return v_col, flux_E


def get_coulomb_scattering_rate(n, Ti, Te, settings, species='ions'):
    """
    Coulomb scattering rate for ions or electrons, scattering off the other species
    n is ni=ne, not the total density n_tot = 2*n
    """
    if settings['assume_constant_density'] is True:
        n = settings['n0'] + 0 * n

    if settings['assume_constant_temperature'] is True:
        Ti = settings['Ti_0'] + 0 * Ti
        Te = settings['Te_0'] + 0 * Te

    if species == 'ions':
        scat_rate = get_specific_coulomb_scattering_rate(n, Te, n, Ti, settings, impact_specie='i', target_specie='e') \
                    + get_specific_coulomb_scattering_rate(n, Te, n, Ti, settings, impact_specie='i', target_specie='i')
        return scat_rate * settings['ion_scattering_rate_factor']
    elif species == 'electrons':
        scat_rate = get_specific_coulomb_scattering_rate(n, Te, n, Ti, settings, impact_specie='e', target_specie='e') \
                    + get_specific_coulomb_scattering_rate(n, Te, n, Ti, settings, impact_specie='e', target_specie='i')
        return scat_rate * settings['electron_scattering_rate_factor']
    else:
        raise ValueError('invalid option for species = ' + str(species))


def get_specific_coulomb_scattering_rate(ne, Te, ni, Ti, settings, impact_specie='e', target_specie='i'):
    """
    Coulomb scattering rate based on the general formula of
    "2007 - Fundamenski et al - Comparison of Coulomb Collision Rates in the
    Plasma Physics and Magnetically Confined Fusion Literature"
    densities in m^-3 and temperatures in eV
    """
    eps0 = define_vacuum_permittivity()
    e = define_electron_charge()
    coulomb_log_dict = calculate_coulomb_logarithm(ne, Te, ni, Ti,
                                                   Z=settings['Z_ion'],
                                                   A=settings['A_atomic_weight'],
                                                   coulomb_log_min=settings['coulomb_log_min'])

    if impact_specie == 'e':
        m_s1 = settings['me']
        Z_s1 = 1
        T_s1 = Te
    else:
        m_s1 = settings['mi']
        Z_s1 = settings['Z_ion']
        T_s1 = Ti

    if target_specie == 'e':
        m_s2 = settings['me']
        Z_s2 = 1
        T_s2 = Te
        n_s2 = ne
    else:
        m_s2 = settings['mi']
        Z_s2 = settings['Z_ion']
        T_s2 = Ti
        n_s2 = ni

    if impact_specie == 'e' and target_specie == 'e':
        coulomb_log = coulomb_log_dict['ee']
    elif impact_specie == 'i' and target_specie == 'i':
        coulomb_log = coulomb_log_dict['ii']
    elif (impact_specie == 'e' and target_specie == 'i') or (impact_specie == 'i' and target_specie == 'e'):
        # as a shortcut, check only the first cell temperatures to pick the correct temperature regine
        Ti_0 = settings['Ti_0']
        Te_0 = settings['Te_0']
        if Ti_0 / Te_0 > settings['mi'] / settings['me']:
            coulomb_log = coulomb_log_dict['ie_overheated_ions']
        else:
            if Te_0 > 10 * settings['Z_ion'] ** 2.0:
                coulomb_log = coulomb_log_dict['ie_hot_electrons']
            else:
                coulomb_log = coulomb_log_dict['ie_cold_electrons']

    else:
        raise ValueError('invalid option for impact/target species.')
    scat_rate = 2 ** 0.5 / (12 * np.pi ** 1.5) * n_s2 * Z_s1 ** 2 * Z_s2 ** 2 * e ** 4 * coulomb_log / \
                (eps0 ** 2 * m_s1 ** 0.5 * (e * T_s1) ** 1.5) \
                * (1 + m_s1 / m_s2) / (1 + m_s1 / m_s2 * T_s1 / T_s2) ** 1.5
    return scat_rate


def calculate_coulomb_logarithm(ne, Te, ni, Ti, Z=1, A=1, coulomb_log_min=1):
    """
    Coulomb logarithm for different interactions, based on the NRL formulary 2019.
    densities in m^-3 and temperatures in eV
    """
    ne_cm = ne * 1e-6  # convert units to cm^-3 used in the formulas
    ni_cm = ni * 1e-6
    coulomb_log = {}
    coulomb_log['ee'] = 23.5 - np.log(ne_cm ** 0.5 * Te ** (-5.0 / 4)) \
                        - (1e-5 + (np.log(Te) - 2) ** 2.0 / 16.0) ** 0.5
    coulomb_log['ii'] = 23.0 - np.log(Z ** 3.0 * 2 ** 0.5 * ni_cm ** 0.5 * Ti ** (-3.0 / 2))
    coulomb_log['ie_overheated_ions'] = 16.0 - np.log(Z ** 2.0 * A * ni_cm ** 0.5 * Ti ** (-3.0 / 2))
    coulomb_log['ie_cold_electrons'] = 23.0 - np.log(ne_cm ** 0.5 * Te ** (-1.0))
    coulomb_log['ie_hot_electrons'] = 24.0 - np.log(Z * ne_cm ** 0.5 * Te ** (-3.0 / 2))

    # make sure the coulomb log does not get too low (unphysical parameter range)
    for key in coulomb_log:
        coulomb_log[key] = theta_fun_generic(coulomb_log[key], x0=coulomb_log_min)

    return coulomb_log


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


def get_transmission_velocities(state, settings):
    v_R = state['v_th'] + state['v_col']  # defined positive to the right
    v_L = state['v_th'] - state['v_col']  # defined positive to the left

    if settings['assume_constant_transmission'] is True:
        v_R = v_R[0] + 0 * v_R
        v_L = v_L[0] + 0 * v_L
    else:
        # for high enough collective velocity, left transmission is halted
        for k in range(settings['number_of_cells']):
            if v_L[k] < 0: v_L[k] = 0

        if settings['transition_type'] == 'none':
            pass
        elif settings['transition_type'] == 'smooth_transition_to_free_flow':
            # reduce the left transmission below the threshold, smoothly
            v_L = v_L / state['f_above']
        elif settings['transition_type'] == 'sharp_transition_to_free_flow':
            # reduce the left transmission below the threshold, sharply
            for k in range(settings['number_of_cells']):
                if state['n'][k] < settings['n_transition']: v_L[k] = 0
        else:
            raise TypeError('invalid transition_type = ' + settings['transition_type'])

    return v_R, v_L


def calculate_transition_density(n, Ti, Te, settings, state=None):
    if state is not None and 'mean_free_path' in state:
        mfp = state['mean_free_path']
    else:
        mfp = calculate_mean_free_path(n, Ti, Te, settings, state=state)

    if settings['assume_constant_temperature'] is True:
        return settings['n0'] * (settings['mfp_max'] / mfp) ** (-1)
    else:
        return settings['n0'] * (settings['mfp_min'] / mfp) ** (
                1 / (2 * get_gamma_dimension(settings['plasma_dimension']) - 3))


def get_mmm_velocity(state, settings):
    # define MMM velocity for different cells

    if settings['mmm_velocity_type'] == 'absolute':
        U0 = settings['U0']
    elif settings['mmm_velocity_type'] == 'relative_to_thermal_velocity':
        v_th = state['v_th']
        U0 = settings['U0'] * v_th[0]
    else:
        raise ValueError('invalid mmm_velocity_type: ' + settings['mmm_velocity_type'])

    if settings['adaptive_mirror'] == 'adjust_U':
        # change U at different positions directly. This is only theoretical, because in reality we can define
        # the MMM frequency, and the velocity is derived from the wavelength (U=wavelength*frequency)
        v_th = state['v_th']
        U = U0 * v_th / v_th[0]

    elif settings['adaptive_mirror'] in ['adjust_cell_size_with_mfp', 'adjust_cell_size_with_vth']:
        # the realistic method, the MMM velocity scales with the cell size (wavelength)
        mirror_cell_sizes = state['mirror_cell_sizes']
        U = U0 * mirror_cell_sizes / mirror_cell_sizes[0]
    else:
        U = U0 + 0 * state['n']

    if settings['transition_type'] == 'none':
        pass
    elif settings['transition_type'] == 'smooth_transition_to_free_flow':
        U = U / state['f_above']
    elif settings['transition_type'] == 'sharp_transition_to_free_flow':
        # reduce the left transmission below the threshold, sharply
        for k in range(settings['number_of_cells']):
            if state['n'][k] < settings['n_transition']: U[k] = 0
    else:
        raise ValueError('invalid transition_type: ' + settings['transition_type'])

    return U


def define_loss_cone_fractions(state, settings):
    v_th = state['v_th']
    U = state['U'] * settings['U_for_loss_cone_factor']
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

    # # testing making smaller the part that can get scattered to be trapped
    # alpha_tR *= 2
    # alpha_tL *= 2
    # alpha_c = 1 - alpha_tR - alpha_tL
    # alpha_c = 0.5 - alpha_tR - alpha_tL

    return alpha_tR, alpha_tL, alpha_c


def get_density_time_derivatives(state, settings):
    """
    The core of the relaxation algorithm, define the rate equations that advances the state variables
    """

    # state variables
    n_c = state['n_c']
    n_tL = state['n_tL']
    n_tR = state['n_tR']
    nu_s = state['coulomb_scattering_rate']
    v_R = state['v_R']
    v_L = state['v_L']
    U = state['U']
    cell_sizes = state['mirror_cell_sizes']
    alpha_tL = state['alpha_tL']
    alpha_tR = state['alpha_tR']
    alpha_c = state['alpha_c']

    # define density time derivative
    f_scat_c = + nu_s * (alpha_c * (n_tL + n_tR * settings['right_scat_factor'])
                         - (alpha_tL + alpha_tR) * n_c)

    f_scat_tL = + nu_s * (-(alpha_c + alpha_tR) * n_tL
                          + alpha_tL * n_tR * settings['right_scat_factor']
                          + alpha_tL * n_c)

    f_scat_tR = + nu_s * (-(alpha_c + alpha_tL) * n_tR * settings['right_scat_factor']
                          + alpha_tR * n_tL
                          + alpha_tR * n_c)

    coeff_mat_L = - np.eye(settings['number_of_cells']) + np.eye(settings['number_of_cells'], k=1)
    f_trans_L = coeff_mat_L.dot(v_L * n_tL)
    f_trans_L *= settings['transmission_factor']
    f_trans_L /= cell_sizes

    coeff_mat_R = - np.eye(settings['number_of_cells']) + np.eye(settings['number_of_cells'], k=-1)
    f_trans_R = coeff_mat_R.dot(v_R * n_tR)
    f_trans_R *= settings['transmission_factor']
    f_trans_R /= cell_sizes

    f_drag = coeff_mat_L.dot(U * n_c)
    f_drag /= cell_sizes

    # combine rates
    dn_c_dt = f_scat_c + f_drag
    dn_tL_dt = f_scat_tL + f_trans_L
    dn_tR_dt = f_scat_tR + f_trans_R

    if settings['use_RF_terms'] == True:
        tau_th = settings['cell_size'] / state['v_th']
        nu_RF_cl = settings['RF_capacity_cl'] / tau_th
        nu_RF_cr = settings['RF_capacity_cr'] / tau_th
        nu_RF_lc = settings['RF_capacity_lc'] / tau_th
        nu_RF_rc = settings['RF_capacity_rc'] / tau_th

        f_RF_c = - (nu_RF_cl + nu_RF_cr) * n_c + nu_RF_lc * n_tL + nu_RF_rc * n_tR
        f_RF_tL = - nu_RF_lc * n_tL + nu_RF_cl * n_c
        f_RF_tR = - nu_RF_rc * n_tR + nu_RF_cr * n_c

        dn_c_dt += f_RF_c
        dn_tL_dt += f_RF_tL
        dn_tR_dt += f_RF_tR

    return dn_c_dt, dn_tL_dt, dn_tR_dt


def get_fluxes(state, settings):
    # state variables
    n_c = state['n_c']
    n_tL = state['n_tL']
    n_tR = state['n_tR']
    v_R = state['v_R']
    v_L = state['v_L']
    U = state['U']

    # initializations
    flux_trans_R = np.nan * np.zeros(settings['number_of_cells'])
    flux_trans_L = np.nan * np.zeros(settings['number_of_cells'])
    flux_mmm_drag = np.nan * np.zeros(settings['number_of_cells'])

    # calculate fluxes (normalized to the single mirror flux)
    for k in range(0, settings['number_of_cells'] - 1):
        flux_trans_R[k] = v_R[k] * n_tR[k]
        flux_trans_L[k] = - v_L[k + 1] * n_tL[k + 1]
        flux_mmm_drag[k] = - U[k + 1] * n_c[k + 1]
    flux_single = state['n'][0] * state['v_th'][0]

    flux = flux_trans_R + flux_trans_L + flux_mmm_drag

    # save fluxes to state
    state['flux_trans_R'] = flux_trans_R / flux_single
    state['flux_trans_L'] = flux_trans_L / flux_single
    state['flux_mmm_drag'] = flux_mmm_drag / flux_single
    state['flux'] = flux / flux_single

    # calculate flux  statistics
    state['flux_max'] = np.nanmax(flux)
    state['flux_min'] = np.nanmin(flux)
    state['flux_mean'] = np.nanmean(flux)
    state['flux_std'] = np.nanstd(flux)
    state['flux_normalized_std'] = np.abs(state['flux_std'] / state['flux_mean'])

    logging.info('flux_mean = ' + '{:.2e}'.format(state['flux_mean'])
                 + ', flux_normalized_std = ' + '{:.2e}'.format(state['flux_normalized_std']))

    return state
