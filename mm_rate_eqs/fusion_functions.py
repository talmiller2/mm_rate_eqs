import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.integrate import quad

from mm_rate_eqs.constants_functions import define_electron_charge, define_proton_mass, \
    define_fine_structure_constant, define_speed_of_light, define_factor_eV_to_K, define_barn
from mm_rate_eqs.plasma_functions import get_brem_radiation_loss


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
        return sum([branching_ratio * get_sigma_v_fusion(T, reaction=reaction_branch)
                    for reaction_branch, branching_ratio in zip(reactions, branching_ratios)])
    else:
        raise ValueError('invalid reaction: ' + reaction)

    zeta = 1 - (C2 * T + C4 * T ** 2 + C6 * T ** 3) / (1 + C3 * T + C5 * T ** 2 + C7 * T ** 3)
    ksi = C0 * T ** (-1.0 / 3)
    sigma_non_resonance = C1 * zeta ** (-5.0 / 6) * ksi ** 2 * np.exp(-3 * zeta ** (1.0 / 3) * ksi)
    if fit_type == 'Nevins_Swain' and use_resonance == True:  # add a term for resonance
        sigma_resonance = 5.41e-15 * T ** (-3.0 / 2) * np.exp(-147 / T)
    else:
        sigma_resonance = 0
    sigma_v_cm3_over_s = sigma_non_resonance + sigma_resonance
    sigma_v_m3_over_s = 1e-6 * sigma_v_cm3_over_s

    return sigma_v_m3_over_s


def get_fusion_data_file_path(reaction='D_T_to_n_alpha'):
    file = os.path.dirname(__file__) + '/fusion_reactivity_data/' + reaction + '.csv'
    return file


def get_sigma_v_fusion_sampled(T, reaction='D_T_to_n_alpha'):
    """
    Sampling of plots of <sigma*v> instead of using analytic fits.
    Most from Atzeni2004 Book, except the separate D-D reactions from Wurzel2022.
    T in [keV], reactivity sigma*v in [m3/s].
    """

    if reaction == 'p_B_to_3alpha':  # change default data set for pB
        reaction += '_Sikora2016newdata'
        file_path = get_fusion_data_file_path(reaction=reaction)
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        T_data = data[:, 0]
        sigma_v_data = data[:, 1] * 1e-6  # change units [cm^3/s] to [m^3/s]
        interp_fun = interp1d(T_data, sigma_v_data, bounds_error=False)
        sigma_v_interped = interp_fun(T)

    else:
        file_path = get_fusion_data_file_path(reaction=reaction)
        data = np.loadtxt(file_path, delimiter=',')
        log10_T_list = data[:, 0]
        log10_sigma_v_list = data[:, 1]
        interp_fun = interp1d(log10_T_list, log10_sigma_v_list, kind='cubic', bounds_error=False)
        sigma_v_interped = 10.0 ** (interp_fun(np.log10(T)))

    return sigma_v_interped


def load_sigma_v_fusion_files(Ti_keV):
    sigma_v_dict = {}
    reactions = ['D_T_to_n_alpha', 'D_He3_to_p_alpha', 'D_D_to_p_T', 'D_D_to_n_He3', 'p_B_to_3alpha']
    for reaction in reactions:
        sigma_v_dict[reaction] = get_sigma_v_fusion_sampled(Ti_keV, reaction=reaction)
    return sigma_v_dict

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
    elif reaction == 'cat_D_D_to_p_T_n_He3':
        label = 'cat $D + D \\rightarrow p + T, \, n + {}^{3}He$'
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


def update_ion_latex_name(name):
    """
    Return ion names string in latex style.
    """
    name = name.replace('He3', '$^3$He')
    name = name.replace('B11', '$^{11}$B')
    name = name.replace('fully-catalyzed', 'cat')
    return name

def get_E_reaction(reaction='D_T_to_n_alpha'):
    """
    Fusion reaction total output energy, in [MeV].
    Source https://en.wikipedia.org/wiki/Nuclear_fusion.
    """
    # controlled fusion fuels
    if reaction == 'D_T_to_n_alpha':
        E_reaction = 17.6  # 3.5 + 14.1
    elif reaction == 'D_D_to_p_T':
        E_reaction = 4.03  # 3.02 + 1.01
    elif reaction == 'D_D_to_n_He3':
        E_reaction = 3.27  # 2.45 + 0.82
    elif reaction == 'D_D_to_p_T_n_He3':
        E_reaction = 4.03 + 3.27
    elif reaction == 'T_T_to_alpha_2n':
        E_reaction = 11.3
    # advanced controlled fusion fuels
    elif reaction == 'D_He3_to_p_alpha':
        E_reaction = 18.3  # 14.7 + 3.6
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
        E_charged = 0.82
    elif reaction == 'D_D_to_p_T_n_He3':
        E_charged = 0.5 * 3.02 + 0.5 * 0.82
    elif reaction == 'T_T_to_alpha_2n':
        E_charged = 0  # ???
    # advanced controlled fusion fuels
    elif reaction == 'D_He3_to_p_alpha':
        # E_charged = 14.7 # TODO: this was the version up to 31.1.23, it is wrong and missing 3.6
        E_charged = 18.3  # 14.7 + 3.6
    elif reaction == 'p_B_to_3alpha':
        E_charged = 8.7
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


def get_fusion_sigma_v_E_reaction(T, reaction='D_T_to_n_alpha', energy_type='all'):
    """
    Multiply sigma*v by the reaction energy and return in [J].
    T in [keV].
    """
    if energy_type == 'all':
        E_fun = get_E_reaction
    elif energy_type == 'charged':
        E_fun = get_E_charged
    else:
        raise ValueError('invalid energy energy_type=', energy_type)

    if reaction == 'D_D_to_p_T_n_He3':
        reactions = ['D_D_to_p_T', 'D_D_to_n_He3']
        branching_ratios = [0.5, 0.5]
        return sum([branching_ratio * get_sigma_v_fusion(T, reaction=reaction_branch)
                    * E_fun(reaction=reaction_branch)
                    for reaction_branch, branching_ratio in zip(reactions, branching_ratios)])

    else:
        sigma_v_E = get_sigma_v_fusion(T, reaction=reaction) * E_fun(reaction=reaction)
    MeV_to_J = define_electron_charge() * 1e6
    return sigma_v_E * MeV_to_J


def get_fusion_power(ni, T, reaction='D_T_to_n_alpha'):
    """
    Fusion power in [W/m^3], assuming 50-50 split of reacting ions.
    T in [keV], ni in [m^-3].
    """
    return (ni / 2.0) ** 2 * get_fusion_sigma_v_E_reaction(T, reaction=reaction)


def get_fusion_charged_power(ni, T, reaction='D_T_to_n_alpha'):
    """
    Fusion power (only from charged particles) in [W/m^3], assuming 50-50 split of reacting ions.
    T in [keV], ni in [m^-3].
    """
    return (ni / 2.0) ** 2 * get_fusion_sigma_v_E_reaction(T, reaction=reaction, energy_type='charged')


def get_fusion_power_multiple_ions(ni_array, Ti_keV, ions_list, sigma_v_dict=None):
    """
    Fusion power (total and charged) in [W/m^3], for multiple ion species with difference densities.
    Ti_keV in [keV], ni_array in [m^-3], each row is a different ion.
    """
    if sigma_v_dict == None:
        sigma_v_dict = load_sigma_v_fusion_files(Ti_keV)

    e = define_electron_charge()
    MeV_to_J = e * 1e6
    reaction_rate_tot = 0 * Ti_keV
    P_fus_tot = 0 * Ti_keV
    P_fus_charged_tot = 0 * Ti_keV
    for ind_r_1, reactant_1 in enumerate(ions_list):
        for ind_r_2, reactant_2 in enumerate(ions_list):
            if ind_r_2 >= ind_r_1:  # avoid double counting

                # number of pair combinations for same-ion reaction
                if reactant_1 == reactant_2:
                    combinatoric_factor = 0.5
                else:
                    combinatoric_factor = 1

                # by default some processes are neglected (small or no available data)
                if (reactant_1 == 'D' and reactant_2 == 'T') or (reactant_1 == 'T' and reactant_2 == 'D'):
                    reactions = ['D_T_to_n_alpha']
                elif reactant_1 == 'D' and reactant_2 == 'D':
                    # no need to use branching ratios, the cross-sections per reaction already take care of that
                    reactions = ['D_D_to_p_T', 'D_D_to_n_He3']
                elif (reactant_1 == 'D' and reactant_2 == 'He3') or (reactant_1 == 'He3' and reactant_2 == 'D'):
                    reactions = ['D_He3_to_p_alpha']
                elif (reactant_1 == 'p' and reactant_2 == 'B11') or (reactant_1 == 'B11' and reactant_2 == 'p'):
                    reactions = ['p_B_to_3alpha']
                else:
                    reactions = []

                for reaction in reactions:
                    sigma_v_curr = sigma_v_dict[reaction]
                    ni_1_curr = ni_array[ind_r_1, :]
                    ni_2_curr = ni_array[ind_r_2, :]
                    reaction_rate_curr = combinatoric_factor * ni_1_curr * ni_2_curr * sigma_v_curr
                    E_f_curr = get_E_reaction(reaction=reaction)  # [MeV]
                    E_ch_curr = get_E_charged(reaction=reaction)  # [MeV]
                    P_fus_curr = reaction_rate_curr * E_f_curr * MeV_to_J
                    P_fus_charged_curr = reaction_rate_curr * E_ch_curr * MeV_to_J
                    reaction_rate_tot += reaction_rate_curr
                    P_fus_tot += P_fus_curr
                    P_fus_charged_tot += P_fus_charged_curr

    return P_fus_tot, P_fus_charged_tot, reaction_rate_tot


def set_ion_densities_quasi_neutral(process, ne, Ti_keV, sigma_v_dict=None):
    """
    define the ion densities (multiple species) for different processes to satisfy quasi-neutrality with electron density,
    Ti_keV in [keV], ne in [m^-3].
    """
    if sigma_v_dict == None:
        sigma_v_dict = load_sigma_v_fusion_files(Ti_keV)

    if process == 'D-T':
        ions_list = ['D', 'T']
        Zi_list = [1, 1]
        # for const ne, optimal choice to maximize P_fus is (1/2Z), proof in Wurzel2022
        ni_rel_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zi_list]

    elif process == 'D-He3':
        ions_list = ['D', 'He3']
        Zi_list = [1, 2]
        ni_rel_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zi_list]

    elif process == 'p-B11':
        ions_list = ['p', 'B11']
        Zi_list = [1, 5]
        ni_rel_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zi_list]

    elif process == 'pure D-D':
        ions_list = ['D']
        Zi_list = [1]
        ni_rel_list = [1 + 0 * Ti_keV]

    elif process == 'fully-catalyzed D-D':
        # from Wurzel2022: "Here, we only consider the steady-state reaction path where He3 and T react with D at the
        # same rate as they are created in each branch of the D-D reaction. Furthermore, we assume an idealized scenario
        # without synchrotron radiation and that the “ash” alpha particles and protons immediately exit after depositing
        # their energy and comprise a negligible fraction of ions in the plasma. Finally, we assume that D is added at
        # the same rate as it is consumed." See Eqs. (C13)-(C14).
        ions_list = ['D', 'T', 'He3']
        Zi_list = [1, 1, 2]
        ni_rel_list = [1 + 0 * Ti_keV,
                       0.5 * sigma_v_dict['D_D_to_p_T'] / sigma_v_dict['D_T_to_n_alpha'],
                       0.5 * sigma_v_dict['D_D_to_n_He3'] / sigma_v_dict['D_He3_to_p_alpha']]


    elif process == 'He3-catalyzed D-D':
        ions_list = ['D', 'He3']
        Zi_list = [1, 2]
        # densities chosen at steady state for D,He3 (T assumed as extracted instantly)
        ni_rel_list = [1 + 0 * Ti_keV,
                       0.5 * sigma_v_dict['D_D_to_n_He3'] / sigma_v_dict['D_He3_to_p_alpha']]

    elif process == 'T-catalyzed D-D':
        ions_list = ['D', 'T']
        Zi_list = [1, 1]
        # densities chosen at steady state for D,T (He3 assumed as extracted instantly)
        ni_rel_list = [1 + 0 * Ti_keV,
                       0.5 * sigma_v_dict['D_D_to_p_T'] / sigma_v_dict['D_T_to_n_alpha']]

    else:
        raise ValueError('invalid process', process)

    # normalize ion densities to satisfy quasi-neutrality with electrons
    ni_rel_array = np.array(ni_rel_list)
    neutrality_fac = sum([ni_rel * Zi for ni_rel, Zi in zip(ni_rel_array, Zi_list)])
    ni_array = 0 * ni_rel_array
    for ind_ion in range(ni_rel_array.shape[0]):
        ni_array[ind_ion, :] = ni_rel_array[ind_ion, :] * ne / neutrality_fac
    ni = np.sum(ni_array, axis=0)

    return ions_list, Zi_list, ni_array, ni

def get_lawson_parameters(ni, Ti, settings, reaction='D_T_to_n_alpha'):
    """
    Lawson minimal confinement time [s], and maximal flux [s^-1] from each end assuming some fusion volume.
    T in [keV], ni in [m^-3].
    """
    sigma_v_fusion = get_sigma_v_fusion(Ti, reaction=reaction)
    MeV_to_J = define_electron_charge() * 1e6
    E_charged = get_E_charged(reaction=reaction) * MeV_to_J  # J
    kB_keV = define_electron_charge() * 1e3
    n_tau_lawson = 12 * kB_keV * Ti / (E_charged * sigma_v_fusion)
    tau_lawson = n_tau_lawson / ni
    # flux_lawson = 0.5 / n_tau_lawson * settings['volume_main_cell'] * ni ** 2
    flux_lawson = 0.5 * ni * settings['volume_main_cell'] / tau_lawson
    return tau_lawson, flux_lawson


def get_lawson_criterion_piel(ni, Ti, settings, eta=0.3, reaction='D_T_to_n_alpha'):
    """
    Lawson minimal confinement time [s] based on derivation in Piel (2007) book, page 105,
    which is made specifically for 50%-50% D-T mix, but we can generalize for other 50%-50% mixtures.
    Paraphrasing, the energy balance is: P_br + P_H = eta * (P_br + P_H + P_fus), from which we can
    extract the Lawson criterion for different heat-electricity efficiencies eta (0.3 as an example).
    Ignition is when: P_br + P_H = P_ch.
    Plugging ignition condition to the balance eq. gives eta = P_ch / (P_fus + P_ch),
    which is 0.166 for DT, but Piel writes 0.154.
    Input: T in [keV], ni in [m^-3].
    """
    sigma_v_fusion = get_sigma_v_fusion(Ti, reaction=reaction)
    MeV_to_J = define_electron_charge() * 1e6
    E_charged = get_E_charged(reaction=reaction) * MeV_to_J  # J
    E_reaction = get_E_reaction(reaction=reaction) * MeV_to_J  # J
    kB_keV = define_electron_charge() * 1e3
    P_brem_vol = get_brem_radiation_loss(ni, ni, Ti, settings['Z_ion'])  # W/m^3

    tau_lawson_piel = 3 * kB_keV * Ti / ni / (
            eta / (1 - eta) * 0.25 * E_reaction * sigma_v_fusion - P_brem_vol / ni ** 2)
    flux_lawson_piel = 0.5 * ni * settings['volume_main_cell'] / tau_lawson_piel

    eta_ignition = E_charged / (E_charged + E_reaction)
    tau_lawson_ignition_piel = 3 * kB_keV * Ti / ni / (
            eta_ignition / (1 - eta_ignition) * 0.25 * E_reaction * sigma_v_fusion - P_brem_vol / ni ** 2)
    flux_lawson_ignition_piel = 0.5 * ni * settings['volume_main_cell'] / tau_lawson_ignition_piel

    return tau_lawson_piel, flux_lawson_piel, tau_lawson_ignition_piel, flux_lawson_ignition_piel


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
    elif reaction == 'D_D_to_p_T_n_He3':
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
    Return Gamow energy (tunneling barrier) in [keV].
    Source: "Atzeni, Meyer-ter-Vehn - The Physics of Inertial Fusion", chapter 1.
    """
    # alpha_fine = 1 / 137.035999084 # fine structure constant
    A_1, A_2 = get_As_for_reaction(reaction=reaction)
    Z_1, Z_2 = get_Zs_for_reaction(reaction=reaction)
    A_r = A_1 * A_2 / (A_1 + A_2)  # reduced A
    # E_g = 986.1 * (Z_1 * Z_2) ** 2 * A_r # shortened form

    m_p = define_proton_mass()  # kg
    m_r = m_p * A_1 * A_2 / (A_1 + A_2)  # reduced mass
    alpha = define_fine_structure_constant()
    e = define_electron_charge()
    c = define_speed_of_light()  # m/s
    m_r_keV = m_r * c ** 2 / e / 1e3
    E_g = (np.pi * alpha * Z_1 * Z_2) ** 2 * 2 * m_r_keV

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
        c_cm = c * 1e2
        m_r_keV_over_c2 = m_r * c ** 2 / e / 1e3
        m_r_keV_cm = m_r_keV_over_c2 / c_cm ** 2

        # effective S factor
        E0 = T * (np.pi * alpha * Z_1 * Z_2 / np.sqrt(2)) ** (2 / 3) * (m_r_keV_over_c2 / T) ** (1 / 3)  # keV
        delta_E0 = T * 4 * np.sqrt(E0 / (3 * T))  # keV
        S0, S0_der, S0_der2 = get_astrophysical_S_factor(reaction=reaction)
        S_eff = S0 * (1 + 5 * T / (36 * E0)) + S0_der * E0 * (1 + 35 * T / (36 * E0)) \
                + 0.5 * S0_der2 * E0 ** 2 * (1 + 89 * T / (36 * E0))  # [keV * barn]
        barn = define_barn()
        barn_cm = barn * 1e4
        S_eff_cm2 = S_eff * barn_cm

        # electron screening factor
        if n is None:
            f0 = 1  # assume no electron screening
        else:
            zeta = ((0.5 * Z_1 ** 2 / A_1 + 0.5 * Z_2 ** 2 / A_2) + 0.92 * (0.5 * Z_1 / A_1 + 0.5 * Z_2 / A_2)) ** 0.5
            rho_kg_over_cm3 = n * m_r  # specific density [kg/cm^3] since n in [cm^3]
            rho0 = rho_kg_over_cm3 * 1e3  # [g/cm^3]
            T_eV = T * 1e3
            T_K = T_eV * define_factor_eV_to_K()
            T_6 = T_K / 1e6  # in 10^6 Kelvin
            f0 = np.exp(0.188 * Z_1 * Z_2 * zeta * rho0 ** 0.5 * T_6 ** (-3 / 2))

        sigma_v_cm3_over_s = np.sqrt(2 / (m_r_keV_cm * T)) * delta_E0 / T * f0 * S_eff_cm2 * np.exp(-3 * E0 / T)
        sigma_v_m3_over_s = 1e-6 * sigma_v_cm3_over_s
        return sigma_v_m3_over_s


def get_sigma_v_fusion_numeric_integration(T_keV_array, reaction='D_T_to_n_alpha'):
    """
    Approximation of <sigma*v> (Maxwell averaged reactivity), numerically integrating sigma(E).
    T in [keV], reactivity sigma*v in [m^3/s].
    """

    E_array = T_keV_array
    sigma_array = get_sigma_fusion(T_keV_array, reaction=reaction)
    sigma_interp = interp1d(E_array, sigma_array, kind='cubic', fill_value=0, bounds_error=False)

    A_1, A_2 = get_As_for_reaction(reaction=reaction)
    A_r = A_1 * A_2 / (A_1 + A_2)  # reduced mass

    def prefactor(mu, kT):
        return 1e-6 * (3.72e-16 / np.sqrt(mu)) * (kT ** -1.5)  # m^3/s units

    def integrand(E, kT):
        return E * sigma_interp(E) * np.exp(-E / kT)  # sigma must be in barns here

    def compute_sigmav(kT, mu):
        integral, _ = quad(integrand, 0, 20 * kT, args=(kT,))
        return prefactor(mu, kT) * integral

    sigma_v_m3_over_s = np.array([compute_sigmav(t, mu=A_r) for t in T_keV_array])

    return sigma_v_m3_over_s
