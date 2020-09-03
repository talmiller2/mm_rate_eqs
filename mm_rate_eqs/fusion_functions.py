import numpy as np


def get_sigma_v_fusion(T_eV, reaction='D-T_to_n_alpha'):
    # fit from http://www.fisicanucleare.it/documents/0-19-856264-0.pdf
    if reaction == 'D-T_to_n_alpha':
        C0 = 6.661
        C1 = 643.41e-16
        C2 = 15.136e-3
        C3 = 75.189e-3
        C4 = 4.6064e-3
        C5 = 13.5e-3
        C6 = -0.10675e-3
        C7 = 0.01366e-3
    elif reaction == 'D-D_to_p_T':
        C0 = 6.2696
        C1 = 3.7212e-16
        C2 = 3.4127e-3
        C3 = 1.9917
        C4 = 0
        C5 = 0.010506
        C6 = 0
        C7 = 0
    elif reaction == 'D-D_to_n_He3':
        C0 = 6.2696
        C1 = 3.5741e-16
        C2 = 8.8577e-3
        C3 = 7.6822e-3
        C4 = 0
        C5 = -0.002964e-3
        C6 = 0
        C7 = 0
    elif reaction == 'He3-D_to_p_alpha':
        C0 = 10.572
        C1 = 151.16e-16
        C2 = 6.4192e-3
        C3 = -2.029e-3
        C4 = -0.019108e-3
        C5 = 0.13578e-3
        C6 = 0
        C7 = 0
    elif reaction == 'p_B_to_3alpha':
        C0 = 17.708
        C1 = 6382e-16
        C2 = -59.357e-3
        C3 = 201.65e-3
        C4 = 1.0404e-3
        C5 = 2.7621e-3
        C6 = -0.0091653e-3
        C7 = 0.00098305e-3
    elif reaction == 'D-D_to_p_T_n_He3':
        pass
    else:
        raise ValueError('invalid reaction: ' + reaction)

    if reaction == 'D-D_to_p_T_n_He3':
        sigma_v_m3_over_s = get_sigma_v_fusion(T_eV, reaction='D-D_to_p_T') \
                            + get_sigma_v_fusion(T_eV, reaction='D-D_to_n_He3')
    else:
        T = T_eV * 1e-3  # T in keV
        zeta = 1 - (C2 * T + C4 * T ** 2 + C6 * T ** 3) / (1 + C3 * T + C5 * T ** 2 + C7 * T ** 3)
        ksi = C0 * T ** (-1.0 / 3)
        sigma_v_cm3_over_s = C1 * zeta ** (-5.0 / 6) * ksi ** 2 * np.exp(-3 * zeta ** (1.0 / 3) * ksi)
        sigma_v_m3_over_s = 1e-6 * sigma_v_cm3_over_s

    return sigma_v_m3_over_s


def get_E_reaction(reaction='D-T_to_n_alpha'):
    # in MeV, from https://en.wikipedia.org/wiki/Nuclear_fusion
    if reaction == 'D-T_to_n_alpha':
        E_reaction = 17.6
    elif reaction == 'D-D_to_p_T':
        E_reaction = 4.03
    elif reaction == 'D-D_to_n_He3':
        E_reaction = 3.27
    elif reaction == 'D-D_to_p_T_n_He3':
        E_reaction = 4.03 + 3.27
    elif reaction == 'He3-D_to_p_alpha':
        E_reaction = 18.2
    elif reaction == 'p_B_to_3alpha':
        E_reaction = 8.7
    else:
        raise ValueError('invalid reaction: ' + reaction)
    return E_reaction


def get_E_charged(reaction='D-T_to_n_alpha'):
    # in MeV, from https://en.wikipedia.org/wiki/Nuclear_fusion
    if reaction == 'D-T_to_n_alpha':
        E_charged = 3.5
    elif reaction == 'D-D_to_p_T':
        E_charged = 3.02
    elif reaction == 'D-D_to_n_He3':
        E_charged = 0
    elif reaction == 'D-D_to_p_T_n_He3':
        E_charged = 3.02
    elif reaction == 'He3-D_to_p_alpha':
        E_charged = 14.7
    elif reaction == 'p_B_to_3alpha':
        E_charged = 0
    else:
        print('invalid reaction: ' + reaction)
    return E_charged


def get_fusion_sigma_v_E_reaction(T_eV, reaction='D-T_to_n_alpha'):
    if reaction == 'D-D_to_p_T_n_He3':
        sigma_v_E = get_sigma_v_fusion(T_eV, reaction='D-D_to_p_T') * get_E_reaction(reaction='D-D_to_p_T') \
                    + get_sigma_v_fusion(T_eV, reaction='D-D_to_n_He3') * get_E_reaction(reaction='D-D_to_n_He3')
    else:
        sigma_v_E = get_sigma_v_fusion(T_eV, reaction=reaction) * get_E_reaction(reaction=reaction)
    return sigma_v_E * 1e6 * 1.6e-19  # MeV to J


def get_fusion_power(ni, T_keV, reaction='D-T_to_n_alpha'):
    return (ni / 2.0) ** 2 * get_fusion_sigma_v_E_reaction(T_keV * 1e3, reaction=reaction)


def get_fusion_charged_power(ni, T_keV, reaction='D-T_to_n_alpha'):
    return (ni / 2.0) ** 2 * get_sigma_v_fusion(T_keV * 1e3, reaction=reaction) \
           * get_E_charged(reaction=reaction) * 1e6 * 1.6e-19  # MeV to J


def get_brem_radiation_loss(ni, ne, Te, Z_ion):
    """
    Bremsstrahlung radiation (source "Fusion Plasma Analysis", p. 228)
    input T in [keV], n in [m^-3] (mks)
    output in [W/m^3]
    """
    return 4.8e-37 * Z_ion ** 2 * ni * ne * Te ** (0.5)


def get_cyclotron_radiation_loss(ne, Te, B):
    """
    Cyclotron/synchrotron radiation (source "Fusion Plasma Analysis", p. 231)
    Majority self-absorbs to only 1e-2 of it remain (source Wesson "Tokamaks" p. 230)
    input T in [keV], n in [m^-3] (mks)
    output in [W/m^3]
    """
    cyclotron_power = 6.2e-17 * B ** 2 * ne * Te
    radiated_fraction = 1e-2
    return radiated_fraction * cyclotron_power


def get_lawson_parameters(ni, Ti, settings, reaction='D-T_to_n_alpha'):
    sigma_v_fusion = get_sigma_v_fusion(Ti, reaction=reaction)
    E_charged = get_E_charged(reaction=reaction) * settings['MeV_to_J']  # J
    n_tau_lawson = 12 * settings['kB_eV'] * Ti / (E_charged * sigma_v_fusion)
    tau_lawson = n_tau_lawson / ni
    flux_lawson = 1 / n_tau_lawson * settings['volume_main_cell'] * ni ** 2
    return tau_lawson, flux_lawson


def define_plasma_parameters(gas_name='hydrogen', ionization_level=1):
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
        A = np.mean([2.01410177811, 3.0160492])  # approximate as mean of D and T
        Z = 1.0
    elif gas_name == 'helium':
        A = 4.002602
        Z = 2.0
    elif gas_name == 'lithium':
        A = 6.941  # 92.41% Li7 A=7.016, 7.59% Li6 A=6.015 (Wikipedia)
        Z = 3.0
    elif gas_name == 'sodium':
        A = 22.9897
        Z = 11.0
    elif gas_name == 'potassium':
        A = 39.0983
        Z = 19.0
    else:
        raise TypeError('invalid gas: ' + gas_name)
    mi = A * mp
    # for non-fusion experiments with low temperature, the ions are not fully ionized
    if ionization_level is not None:
        Z = ionization_level
    return me, mp, mi, A, Z


def get_debye_length(n, Te):
    """
    scale above which quasi-neutrality holds, dominated by the fast electrons.
    From Bellan 'Funamentals of Plasma Physics' p. 9, 20
    n in [m^-3], Te in [keV], return in [m]
    """
    return 0.76e-4 * np.sqrt(Te / 5.0 / (n / 1e20))


def get_larmor_radius(Ti, B, gas_name='hydrogen', ionization_level=None):
    """
    Gyration radius, dominated by the heavy ions
    source https://en.wikipedia.org/wiki/Gyroradius
    Ti in [keV], B in [Tesla], return in [m]
    """
    electron_gyration_radius = 2.2e-5 * np.sqrt(Ti / 5.0) / (B / 1.0)
    me, mp, mi, A, Z = define_plasma_parameters(gas_name=gas_name, ionization_level=ionization_level)
    ion_gyration_radius = np.sqrt(mp / me) * np.sqrt(A) / Z * electron_gyration_radius
    return ion_gyration_radius


def get_magnetic_pressure(B):
    """
    Magnetic pressure B^2/(2*mu0)
    source https://en.wikipedia.org/wiki/Magnetic_pressure
    B in [Tesla], return in [bar]
    """
    return (B / 0.501) ** 2.0

def get_magnetic_field_for_given_pressure(P, beta=1.0):
    """
    Calculate the magnetic field associated with some beta value for a given pressure
    Inverse of get_magnetic_pressure function.
    P in [bar], return in [Tesla]
    """
    return 0.501 * (P / beta) ** 0.5

def get_ideal_gas_pressure(n, T, settings):
    """
    Ideal gas pressure kB*n*T
    source https://en.wikipedia.org/wiki/Boltzmann_constant
    n in total density [m^-3], T in [eV], return in [bar]
    """
    Pa_to_bar = 1e-5
    return settings['kB_K'] * n * settings['eV_to_K'] * T * Pa_to_bar


def get_ideal_gas_energy_per_volume(n, T, settings):
    """
    Ideal gas energy for monoatomic gas 3/2*kB*n*T
    n in total density [m^-3], T in [eV], return in [J/m^3]=[bar]
    """
    return 3.0 / 2 * settings['kB_K'] * n * settings['eV_to_K'] * T
