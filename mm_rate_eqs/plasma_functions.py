import numpy as np

from mm_rate_eqs.constants_functions import define_electron_mass, define_proton_mass, define_factor_eV_to_K, \
    define_boltzmann_constant, define_factor_Pa_to_bar, define_vacuum_permeability, define_electron_charge, \
    define_vacuum_permittivity, define_electron_mass_keV


def define_plasma_parameters(gas_name='hydrogen', ionization_level=None):
    me = define_electron_mass()
    mp = define_proton_mass()
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
        if ionization_level <= Z:
            Z = ionization_level
        else:
            raise ValueError('ionization level cannot be larger that the atomic charge Z.')
    return me, mp, mi, A, Z


def get_brem_radiation_loss(ni, ne, Te, Z_ion):
    """
    Bremsstrahlung radiation power
    input T in [keV], n in [m^-3] (mks)
    output in [W/m^3]
    """
    # C_B = 4.8e-37 # source Stacey "Fusion Plasma Analysis", p. 228
    # C_B = 5.35e-37  # source Piel (2007) book, page 105.
    C_B = 5.34e-37  # source Wurzel,Hsu(2022)
    return C_B * Z_ion ** 2 * ni * ne * Te ** (0.5)


def get_brem_radiation_loss_relativistic(ni_list, Zi_list, Te, use_relativistic_correction=True):
    """
    Bremsstrahlung radiation power, version from Wurzel,Hsu(2022) with relativistic correction
    input T in [keV], n in [m^-3] (mks)
    output in [W/m^3]
    """
    if use_relativistic_correction == True:
        me_keV = define_electron_mass_keV()
        t = Te / me_keV
    else:
        t = 0

    C_B = 5.34e-37  # source Wurzel,Hsu(2022)
    ne = sum([nj * Zj for nj, Zj in zip(ni_list, Zi_list)])  # quasi-neutrality
    Z_eff = sum([nj * Zj ** 2 for nj, Zj in zip(ni_list, Zi_list)]) / ne
    gamma_eff = Z_eff * (1 + 1.78 * t ** 1.34) + 2.12 * t * (1 + 1.1 * t + t ** 2.0 - 1.25 * t ** 2.5)
    P_brem = C_B * gamma_eff * Te ** (0.5) * ne ** 2
    return P_brem

def get_cyclotron_radiation_loss(ne, Te, B):
    """
    Cyclotron/synchrotron radiation (source Stacey "Fusion Plasma Analysis", p. 231)
    Majority self-absorbs so only 1e-2 of it escapes (source Wesson "Tokamaks" p. 230)
    input T in [keV], n in [m^-3] (mks), B in [T] (mks)
    output in [W/m^3]
    """
    cyclotron_power = 6.2e-17 * B ** 2 * ne * Te
    radiated_fraction = 1e-2
    return radiated_fraction * cyclotron_power


def get_debye_length(n, Te):
    """
    scale above which quasi-neutrality holds, dominated by the fast electrons.
    From Bellan 'Fundamentals of Plasma Physics' p. 9, 20
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


def get_larmor_frequency(B, gas_name='hydrogen', ionization_level=None):
    """
    Ti in [keV], B in [Tesla], return in [m]
    """
    e = define_electron_charge()
    if ionization_level is not None:
        qi = ionization_level * e
    else:
        qi = e
    me, mp, mi, A, Z = define_plasma_parameters(gas_name=gas_name, ionization_level=ionization_level)
    omega_cyclotron = qi * B / mi
    f_cyclotron = omega_cyclotron / (2 * np.pi)
    return f_cyclotron


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
    kB = define_boltzmann_constant()
    return kB * n * T * define_factor_eV_to_K() * define_factor_Pa_to_bar()


def get_ideal_gas_energy_per_volume(n, T, settings):
    """
    Ideal gas energy for monoatomic gas 3/2*kB*n*T
    n in total density [m^-3], T in [eV], return in [J/m^3]=[bar]
    """
    kB = define_boltzmann_constant()
    return 3.0 / 2 * kB * n * T * define_factor_eV_to_K()


def get_bohm_diffusion_constant(Te, B):
    """
    Bohm diffusion coefficient D_bohm = kB*T/e / 16B. Note the units [kB*T/e]=[Volts]
    source https://en.wikipedia.org/wiki/Bohm_diffusion
    Te is the electron temperature in [eV], B the magnetic field in [Tesla]=[Volt*s/m^2]
    The diffusion coefficient output is in units [Volt/Tesla]=[m^2/s]
    """
    return Te / B / 16


def get_alfven_wave_group_velocity(B, ni, gas_name='hydrogen', ionization_level=None):
    """
    Calculate the Alfven wave group velocity
    source https://en.wikipedia.org/wiki/Alfv%C3%A9n_wave
    n is ion number density [m^-3], B in [T], return in [m/s]
    """
    mu0 = define_vacuum_permeability()
    me, mp, mi, A, Z = define_plasma_parameters(gas_name=gas_name, ionization_level=ionization_level)
    rho_ions = mi * ni  # ions specific density
    return B / np.sqrt(mu0 * rho_ions)


def get_electron_plasma_frequency(ne):
    """
    Calculate the electron plasma frequency
    n is ion number density [m^-3], return in [m/s] # TODO: Hz?
    """
    eps0 = define_vacuum_permittivity()
    e = define_electron_charge()
    me = define_electron_mass()
    return np.sqrt(ne * e ** 2 / (eps0 * me))


def get_ion_plasma_frequency(ni, gas_name='hydrogen', ionization_level=None):
    """
    Calculate the electron plasma frequency
    n is ion number density [m^-3], return in [m/s]
    """
    eps0 = define_vacuum_permittivity()
    e = define_electron_charge()
    me, mp, mi, A, Z = define_plasma_parameters(gas_name=gas_name, ionization_level=ionization_level)
    return np.sqrt(ni * (e * Z) ** 2 / (eps0 * mi))
