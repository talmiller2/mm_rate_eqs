import numpy as np

from mm_rate_eqs.constants_functions import define_electron_charge, define_electron_mass, define_proton_mass, \
    define_fine_structure_constant, define_speed_of_light, define_factor_eV_to_K


def get_sigma_v_fusion(T, reaction='D_T_to_n_alpha', use_resonance=True):
    """
    Fit forms of <sigma*v> (Maxwell averaged reactivity).
    Source: "Atzeni, Meyer-ter-Vehn - The Physics of Inertial Fusion", chapter 1.
    T in [keV], reactivity sigma*v in [m3/s].
    """
    fit_type = 'Bosch_Hale'
    if reaction == 'D_T_to_n_alpha':
        C0 = 6.661
        C1 = 643.41e-16
        C2 = 15.136e-3
        C3 = 75.189e-3
        C4 = 4.6064e-3
        C5 = 13.5e-3
        C6 = -0.10675e-3
        C7 = 0.01366e-3
    elif reaction == 'D_D_to_p_T':
        C0 = 6.2696
        C1 = 3.7212e-16
        C2 = 3.4127e-3
        C3 = 1.9917
        C4 = 0
        C5 = 0.010506
        C6 = 0
        C7 = 0
    elif reaction == 'D_D_to_n_He3':
        C0 = 6.2696
        C1 = 3.5741e-16
        C2 = 8.8577e-3
        C3 = 7.6822e-3
        C4 = 0
        C5 = -0.002964e-3
        C6 = 0
        C7 = 0
    elif reaction == 'D_He3_to_p_alpha':
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
        fit_type = 'Nevins_Swain'  # add a term for resonance, only for p-B reaction
    elif reaction == 'D_D_to_p_T_n_He3':
        reactions = ['D_D_to_p_T', 'D_D_to_n_He3']
        branching_ratios = [0.5, 0.5]
        return sum([get_sigma_v_fusion(T, reaction=reaction_branch)
                    for reaction_branch, branching_ratios in zip(reactions, branching_ratios)])
    else:
        raise ValueError('invalid reaction: ' + reaction)

    zeta = 1 - (C2 * T + C4 * T ** 2 + C6 * T ** 3) / (1 + C3 * T + C5 * T ** 2 + C7 * T ** 3)
    ksi = C0 * T ** (-1.0 / 3)
    sigma_non_resonance = C1 * zeta ** (-5.0 / 6) * ksi ** 2 * np.exp(-3 * zeta ** (1.0 / 3) * ksi)
    if fit_type == 'Nevins_Swain' and use_resonance == True:  # add a term for resonance
        sigma_resonance = 5.41e-15 * T ** (-3 / 2) * np.exp(-147 / T)
    else:
        sigma_resonance = 0
    sigma_v_cm3_over_s = sigma_non_resonance + sigma_resonance
    sigma_v_m3_over_s = 1e-6 * sigma_v_cm3_over_s

    return sigma_v_m3_over_s


def get_reaction_label(reaction='D_T_to_n_alpha'):
    """
    Return latex style reaction label for plots.
    """
    # controlled fusion fuels
    if reaction == 'D_T_to_n_alpha':
        label = '$D + T \\rightarrow n + \\alpha$'
    elif reaction == 'D_D_to_p_T':
        label = '$D + D \\rightarrow p + T$'
    elif reaction == 'D_D_to_n_He3':
        label = '$D + D \\rightarrow n + {}^{3}He$'
    elif reaction == 'D_D_to_p_T_n_He3':
        label = '$D + D \\rightarrow p + T, \, n + {}^{3}He$'
    elif reaction == 'T_T_to_alpha_2n':
        label = '$T + T \\rightarrow \\alpha + 2n$'
    # advanced controlled fusion fuels
    elif reaction == 'D_He3_to_p_alpha':
        label = '$D + {}^{3}He \\rightarrow p + \\alpha$'
    elif reaction == 'p_B_to_3alpha':
        label = '$p + {}^{11}B \\rightarrow 3 \\alpha$'
    # pp chain fuels
    elif reaction == 'p_p_to_D_e_nu':
        label = '$p + p \\rightarrow D + e^{+} + \\nu_e $'
    elif reaction == 'p_D_to_He3_gamma':
        label = '$p + D \\rightarrow {}^{3}He + \\gamma $'
    elif reaction == 'He3_He3_to_alpha_2p':
        label = '${}^{3}He + {}^{3}He \\rightarrow \\alpha + 2p $'
    else:
        raise ValueError('invalid reaction: ' + reaction)
    return label


def get_E_reaction(reaction='D_T_to_n_alpha'):
    """
    Fusion reaction total output energy, in [MeV].
    Source https://en.wikipedia.org/wiki/Nuclear_fusion.
    """
    # controlled fusion fuels
    if reaction == 'D_T_to_n_alpha':
        E_reaction = 17.6
    elif reaction == 'D_D_to_p_T':
        E_reaction = 4.03
    elif reaction == 'D_D_to_n_He3':
        E_reaction = 3.27
    elif reaction == 'D_D_to_p_T_n_He3':
        E_reaction = 4.03 + 3.27
    elif reaction == 'T_T_to_alpha_2n':
        E_reaction = 11.3
    # advanced controlled fusion fuels
    elif reaction == 'D_He3_to_p_alpha':
        E_reaction = 18.2
    elif reaction == 'p_B_to_3alpha':
        E_reaction = 8.7
    # pp chain fuels
    elif reaction == 'p_p_to_D_e_nu':
        E_reaction = 1.44  # source: Atzeni
    elif reaction == 'p_D_to_He3_gamma':
        E_reaction = 5.49  # source: Atzeni
    elif reaction == 'He3_He3_to_alpha_2p':
        E_reaction = 12.86  # source: Atzeni
    else:
        raise ValueError('invalid reaction: ' + reaction)
    return E_reaction


def get_E_charged(reaction='D_T_to_n_alpha'):
    """
    Fusion reaction output energy of charged particles only (instantly reabsorbed in plasma), in [MeV].
    Source https://en.wikipedia.org/wiki/Nuclear_fusion.
    """
    # controlled fusion fuels
    if reaction == 'D_T_to_n_alpha':
        E_charged = 3.5
    elif reaction == 'D_D_to_p_T':
        E_charged = 3.02
    elif reaction == 'D_D_to_n_He3':
        E_charged = 0
    elif reaction == 'D_D_to_p_T_n_He3':
        E_charged = 3.02
    elif reaction == 'T_T_to_alpha_2n':
        E_charged = 0  # ???
    # advanced controlled fusion fuels
    elif reaction == 'D_He3_to_p_alpha':
        E_charged = 14.7
    elif reaction == 'p_B_to_3alpha':
        E_charged = 0
    # pp chain fuels
    elif reaction == 'p_p_to_D_e_nu':
        E_charged = 0  # ???
    elif reaction == 'p_D_to_He3_gamma':
        E_charged = 0  # ???
    elif reaction == 'He3_He3_to_alpha_2p':
        E_charged = 0  # ???
    else:
        raise ValueError('invalid reaction: ' + reaction)
    return E_charged


def get_fusion_sigma_v_E_reaction(T, reaction='D_T_to_n_alpha'):
    """
    Multiply sigma*v by the reaction energy and return in J.
    T in [keV].
    """
    if reaction == 'D_D_to_p_T_n_He3':
        sigma_v_E = get_sigma_v_fusion(T, reaction='D_D_to_p_T') * get_E_reaction(reaction='D_D_to_p_T') \
                    + get_sigma_v_fusion(T, reaction='D_D_to_n_He3') * get_E_reaction(reaction='D_D_to_n_He3')
    else:
        sigma_v_E = get_sigma_v_fusion(T, reaction=reaction) * get_E_reaction(reaction=reaction)
    e = define_electron_charge()
    return sigma_v_E * 1e6 * e  # MeV to J


def get_fusion_power(ni, T, reaction='D_T_to_n_alpha'):
    """
    Fusion power reaction rate, assuming 50-50 split of reacting ions.
    T in [keV].
    """
    return (ni / 2.0) ** 2 * get_fusion_sigma_v_E_reaction(T, reaction=reaction)


def get_fusion_charged_power(ni, T, reaction='D_T_to_n_alpha'):
    return (ni / 2.0) ** 2 * get_sigma_v_fusion(T, reaction=reaction) \
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
    Majority self-absorbs so only 1e-2 of it escapes (source Wesson "Tokamaks" p. 230)
    input T in [keV], n in [m^-3] (mks)
    output in [W/m^3]
    """
    cyclotron_power = 6.2e-17 * B ** 2 * ne * Te
    radiated_fraction = 1e-2
    return radiated_fraction * cyclotron_power


def get_lawson_parameters(ni, Ti, settings, reaction='D_T_to_n_alpha'):
    sigma_v_fusion = get_sigma_v_fusion(Ti, reaction=reaction)
    E_charged = get_E_charged(reaction=reaction) * settings['MeV_to_J']  # J
    kB_eV = define_electron_charge()
    n_tau_lawson = 12 * kB_eV * Ti / (E_charged * sigma_v_fusion)
    tau_lawson = n_tau_lawson / ni
    flux_lawson = 1 / n_tau_lawson * settings['volume_main_cell'] * ni ** 2
    return tau_lawson, flux_lawson


def define_plasma_parameters(gas_name='hydrogen', ionization_level=1):
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


# TODO: refactor the plasma functions to different file


def get_Zs_for_reaction(reaction='D_T_to_n_alpha'):
    Z_1, Z_2 = None, None
    # controlled fusion fuels
    if reaction == 'D_T_to_n_alpha':
        Z_1, Z_2 = 1, 1
    elif reaction == 'D_D_to_p_T':
        Z_1, Z_2 = 1, 1
    elif reaction == 'D_D_to_n_He3':
        Z_1, Z_2 = 1, 1
    elif reaction == 'T_T_to_alpha_2n':
        Z_1, Z_2 = 1, 1
    # advanced controlled fusion fuels
    elif reaction == 'D_He3_to_p_alpha':
        Z_1, Z_2 = 1, 2
    elif reaction == 'p_B_to_3alpha':
        Z_1, Z_2 = 1, 5
    # pp chain fuels
    elif reaction == 'p_p_to_D_e_nu':
        Z_1, Z_2 = 1, 1
    elif reaction == 'p_D_to_He3_gamma':
        Z_1, Z_2 = 1, 1
    elif reaction == 'He3_He3_to_alpha_2p':
        Z_1, Z_2 = 2, 2
    else:
        print('invalid reaction: ' + reaction)
    return Z_1, Z_2


def get_As_for_reaction(reaction='D_T_to_n_alpha'):
    A_1, A_2 = None, None
    A_p = 1.008
    A_D = 2.014
    A_T = 3.0160492
    A_He3 = 3.016029
    A_B11 = 11.009
    # controlled fusion fuels
    if reaction == 'D_T_to_n_alpha':
        A_1, A_2 = A_D, A_T
    elif reaction == 'D_D_to_p_T':
        A_1, A_2 = A_D, A_D
    elif reaction == 'D_D_to_n_He3':
        A_1, A_2 = A_D, A_D
    elif reaction == 'T_T_to_alpha_2n':
        A_1, A_2 = A_T, A_T
    # advanced controlled fusion fuels
    elif reaction == 'D_He3_to_p_alpha':
        A_1, A_2 = A_D, A_He3
    elif reaction == 'p_B_to_3alpha':
        A_1, A_2 = A_p, A_B11
    # pp chain fuels
    elif reaction == 'p_p_to_D_e_nu':
        A_1, A_2 = A_p, A_p
    elif reaction == 'p_D_to_He3_gamma':
        A_1, A_2 = A_p, A_D
    elif reaction == 'He3_He3_to_alpha_2p':
        A_1, A_2 = A_He3, A_He3
    else:
        print('invalid reaction: ' + reaction)
    return A_1, A_2


def get_gamow_energy(reaction='D_T_to_n_alpha'):
    """
    Return Gamow energy (tunneling barrier) in [keV]
    Source: "Atzeni, Meyer-ter-Vehn - The Physics of Inertial Fusion", chapter 1.
    """
    # alpha_fine = 1 / 137.035999084 # fine structure constant
    A_1, A_2 = get_As_for_reaction(reaction=reaction)
    Z_1, Z_2 = get_Zs_for_reaction(reaction=reaction)
    A_r = A_1 * A_2 / (A_1 + A_2)  # reduced A
    E_g = 986.1 * (Z_1 * Z_2) ** 2 * A_r
    return E_g


def get_astrophysical_S_factor(reaction='D_T_to_n_alpha'):
    """
    Astrophysical S-factor for fusion reactions, in units [keV * barn].
    Approximated as 2nd order polynomial, good for non-resonant reactions.
    Can change by many orders of magnitude for processes involving the different forces.
    Sources: "Atzeni, Meyer-ter-Vehn - The Physics of Inertial Fusion", chapter 1.
             "2011 - Adelberger et al - Solar fusion cross sections. II. The pp chain and CNO cycles"
    """
    S0, S0_der, S0_der2 = None, None, None
    # controlled fusion fuels
    if reaction == 'D_T_to_n_alpha':
        S0 = 1.2e4
    elif reaction == 'D_D_to_p_T':
        S0 = 56
    elif reaction == 'D_D_to_n_He3':
        S0 = 54
    elif reaction == 'T_T_to_alpha_2n':
        S0 = 138
    # advanced controlled fusion fuels
    elif reaction == 'D_He3_to_p_alpha':
        S0 = 5.9e3
    elif reaction == 'p_B_to_3alpha':
        S0 = 2e5
    # pp chain fuels
    elif reaction == 'p_p_to_D_e_nu':
        S0 = 4.01e-22
        S0_der = 4.49e-24
    elif reaction == 'p_D_to_He3_gamma':
        S0 = 2.14e-4
        S0_der = 5.56e-6
        S0_der2 = 9.3e-9
    elif reaction == 'He3_He3_to_alpha_2p':
        S0 = 5.21e3
        S0_der = -4.9
        S0_der2 = 2.2e-2
    else:
        print('invalid reaction: ' + reaction)

    # in case of missing data
    if S0_der == None:
        S0_der = 0
    if S0_der2 == None:
        S0_der2 = 0

    return S0, S0_der, S0_der2


def get_sigma_fusion(E, reaction='D_T_to_n_alpha'):
    """
    Approximated form of fusion cross section as function of COM energy, applicable for non-resonant reactions.
    input E in [keV], return cross section in [barn]
    Source: "Atzeni, Meyer-ter-Vehn - The Physics of Inertial Fusion", chapter 1, eq 1.21
    """
    if reaction == 'D_D_to_p_T_n_He3':
        reactions = ['D_D_to_p_T', 'D_D_to_n_He3']
        branching_ratios = [0.5, 0.5]
        return sum([branching_ratio * get_sigma_fusion(E, reaction=reaction_branch)
                    for reaction_branch, branching_ratio in zip(reactions, branching_ratios)])
    else:
        S0, S0_der, S0_der2 = get_astrophysical_S_factor(reaction=reaction)
        S_of_E = S0 + S0_der * E + 0.5 * S0_der2 * E ** 2  # 2nd order polynomial approximation
        E_g = get_gamow_energy(reaction=reaction)
        sigma = S_of_E / E * np.exp(-(E_g / E) ** 0.5)
        return sigma


def get_sigma_v_fusion_approx(T, reaction='D_T_to_n_alpha', n=None):
    """
    Approximation of <sigma*v> (Maxwell averaged reactivity), applicable for non-resonant reactions.
    Source:"2011 - Adelberger et al - Solar fusion cross sections. II. The pp chain and CNO cycles", eq. 8.
    T in [keV], n is number density in [cm^3], reactivity sigma*v in [m^3/s].
    """
    if reaction == 'D_D_to_p_T_n_He3':
        reactions = ['D_D_to_p_T', 'D_D_to_n_He3']
        branching_ratios = [0.5, 0.5]
        return sum([branching_ratio * get_sigma_v_fusion_approx(T, reaction=reaction_branch, n=n)
                    for reaction_branch, branching_ratio in zip(reactions, branching_ratios)])
    else:

        # constants
        A_1, A_2 = get_As_for_reaction(reaction=reaction)
        Z_1, Z_2 = get_Zs_for_reaction(reaction=reaction)
        m_p = define_proton_mass()  # kg
        m_r = m_p * A_1 * A_2 / (A_1 + A_2)  # reduced mass
        alpha = define_fine_structure_constant()
        e = define_electron_charge()
        c = define_speed_of_light()  # m/s
        m_r_keV = m_r * c ** 2 / e / 1e3

        # effective S factor
        E0 = T * (np.pi * alpha * Z_1 * Z_2 / np.sqrt(2)) ** (2 / 3) * (m_r_keV / T) ** (1 / 3)  # keV
        delta_E0 = T * 4 * np.sqrt(E0 / (3 * T))  # keV
        S0, S0_der, S0_der2 = get_astrophysical_S_factor(reaction=reaction)
        S_eff = S0 * (1 + 5 * T / (36 * E0)) + S0_der * E0 * (1 + 35 * T / (36 * E0)) \
                + 0.5 * S0_der2 * E0 ** 2 * (1 + 89 * T / (36 * E0))  # [keV * barn]
        S_eff_cm2 = S_eff * 1e-24

        # electron screening factor
        if n is None:
            f0 = 1  # assume no electron screening
        else:
            zeta = ((0.5 * Z_1 ** 2 / A_1 + 0.5 * Z_2 ** 2 / A_2) + 0.92 * (0.5 * Z_1 / A_1 + 0.5 * Z_2 / A_2)) ** 0.5
            rho_kg_over_cm3 = n * m_r  # specific density [kg/cm^3] since n [cm^3]
            rho0 = rho_kg_over_cm3 * 1e3  # [g/cm^3]
            T_eV = T * 1e3
            T_K = T_eV * define_factor_eV_to_K()
            T_6 = T_K / 1e6  # in 10^6 Kelvin
            f0 = np.exp(0.188 * Z_1 * Z_2 * zeta * rho0 ** 0.5 * T_6 ** (-3 / 2))

        sigma_v_cm3_over_s = np.sqrt(2 / (m_r_keV * T)) * delta_E0 / T * f0 * S_eff_cm2 * np.exp(-3 * E0 / T)

        # Note an extra factor of c up front is necessary for the units. It comes from the mass in the lower sqrt.
        c_cm = c * 1e2
        sigma_v_cm3_over_s *= c_cm

        sigma_v_m3_over_s = 1e-6 * sigma_v_cm3_over_s
        return sigma_v_m3_over_s
