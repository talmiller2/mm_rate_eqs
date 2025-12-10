def define_electron_mass():
    m_e = 9.10938356e-31  # kg
    return m_e


def define_proton_mass():
    m_p = 1.67262192e-27  # kg
    return m_p


def define_fine_structure_constant():
    alpha = 1 / 137.035999084  # fine structure constant = e^2/hbar*c
    return alpha


def define_speed_of_light():
    c = 3e8  # [m/s]
    return c


def define_electron_charge():
    # unit of charge [Coulomb], and relates energy units E_J = E_eV * e
    # also kB in units [K/eV]
    e = 1.60217662e-19
    return e


def define_electron_mass_keV():
    me = define_electron_mass()
    c = define_speed_of_light()
    e = define_electron_charge()
    me_keV = me * c ** 2.0 / (e * 1e3)  # electron mass energy in keV [511keV]
    return me_keV


def define_boltzmann_constant():
    kB = 1.380649e-23  # [J/K]
    return kB


def define_factor_eV_to_K():
    e = define_electron_charge()
    kB = define_boltzmann_constant()
    eV_to_K = e / kB  # turns out to be 11604.518020148495
    return eV_to_K


def define_barn():
    barn = 1e-28  # [m^2]
    return barn


def define_factor_Pa_to_bar():
    return 1e-5


def define_vacuum_permittivity():
    eps0 = 8.85418781e-12  # [Farad/m]
    return eps0


def define_vacuum_permeability():
    mu0 = 1.25663706212e-6  # [Henry/m]=[kg*m/s^2/Amp^2]
    return mu0
