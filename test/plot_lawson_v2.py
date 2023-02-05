import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 12
plt.close('all')

from mm_rate_eqs.constants_functions import define_proton_mass, define_electron_mass, define_speed_of_light, \
    define_electron_charge
from mm_rate_eqs.fusion_functions import get_E_reaction, get_E_charged, get_reaction_label, get_sigma_v_fusion, \
    get_sigma_v_fusion_sampled

me = define_electron_mass()
mp = define_proton_mass()
c = define_speed_of_light()
e = define_electron_charge()
me_keV = me * c ** 2.0 / (e * 1e3)  # electron mass energy in keV [511keV]
factor_J_to_keV = 1 / (1e3 * e)
factor_keV_to_J = 1e4 * e

# Ti_keV = np.linspace(1, 1000, 1000)
Ti_keV = np.linspace(1, 200, 1000)
Te_keV = Ti_keV

# aux calc
sigma_v_dict = {}
for reaction in ['D_T_to_n_alpha', 'D_D_to_p_T', 'D_He3_to_p_alpha', 'D_D_to_n_He3']:
    sigma_v_dict[reaction] = get_sigma_v_fusion_sampled(Ti_keV, reaction=reaction)

process_list = []
process_list += ['D-T']
process_list += ['D-He3']
process_list += ['p-B11']
process_list += ['pure D-D']
process_list += ['He3-catalyzed D-D']
process_list += ['T-catalyzed D-D']
process_list += ['fully-catalyzed D-D']

color_list = ['b', 'g', 'r', 'grey', 'k', 'k', 'k']
linestyle_list = ['-', '-', '-', '-', '--', ':', '-']

process_opt_dict = {}

for ip, process in enumerate(process_list):
    color = color_list[ip]
    linestyle = linestyle_list[ip]

    print('*****', process, '*****')

    if process == 'D-T':
        ions_list = ['D', 'T']
        Zj_list = [1, 1]
        nj_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zj_list]  # optimal choice for max fusion in binary reaction

    elif process == 'D-He3':
        ions_list = ['D', 'He3']
        Zj_list = [1, 1]
        nj_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zj_list]  # optimal choice for max fusion in binary reaction

    elif process == 'p-B11':
        ions_list = ['p', 'B11']
        Zj_list = [1, 5]
        nj_list = [1 / (2 * Zj) + 0 * Ti_keV for Zj in Zj_list]  # optimal choice for max fusion in binary reaction

    elif process == 'pure D-D':
        ions_list = ['D']
        Zj_list = [1]
        nj_list = [1 + 0 * Ti_keV]

    elif process == 'fully-catalyzed D-D':
        ions_list = ['D', 'T', 'He3']
        Zj_list = [1, 1, 2]
        # densities chosen at steady state for D,T,He3
        nj_list = [1,
                   0.5 * sigma_v_dict['D_D_to_p_T'] / sigma_v_dict['D_T_to_n_alpha'],
                   0.5 * sigma_v_dict['D_D_to_n_He3'] / sigma_v_dict['D_He3_to_p_alpha']]

    elif process == 'He3-catalyzed D-D':
        ions_list = ['D', 'He3']
        Zj_list = [1, 2]
        # densities chosen at steady state for D,He3 (T assumed as extracted instantly)
        nj_list = [1,
                   0.5 * sigma_v_dict['D_D_to_n_He3'] / sigma_v_dict['D_He3_to_p_alpha']]

    elif process == 'T-catalyzed D-D':
        ions_list = ['D', 'T']
        Zj_list = [1, 1]
        # densities chosen at steady state for D,He3 (T assumed as extracted instantly)
        nj_list = [1,
                   0.5 * sigma_v_dict['D_D_to_p_T'] / sigma_v_dict['D_T_to_n_alpha']]

    else:
        raise ValueError('invalid process', process)

    nj_norm = sum(nj_list)  # normalize
    nj_list = [nj / nj_norm for nj in nj_list]

    ni = sum(nj_list)
    ne = sum([nj * Zj for nj, Zj in zip(nj_list, Zj_list)])

    Z_eff = sum([nj * Zj ** 2 for nj, Zj in zip(nj_list, Zj_list)])
    t = Te_keV / me_keV
    gamma_eff = Z_eff * (1 + 1.78 * t ** 1.34) + 2.12 * t * (1 + 1.1 * t + t ** 2.0 - 1.25 * t ** 2.5)

    # Q = 1e-5
    # Q = 0.5
    # Q = 1
    # Q = 10
    Q = np.inf

    Q_fuel_inv = 1 / Q

    denom_f_term = 0
    P_fus_normalized = 0
    for ind_r_1, reactant_1 in enumerate(ions_list):
        for ind_r_2, reactant_2 in enumerate(ions_list):
            # print(reactant_1, reactant_2)
            # by default some processes are neglected (small or not available sigma_v)
            if (reactant_1 == 'D' and reactant_2 == 'T') or (reactant_1 == 'T' and reactant_2 == 'D'):
                reactions = ['D_T_to_n_alpha']
            elif reactant_1 == 'D' and reactant_2 == 'D':
                reactions = ['D_D_to_p_T', 'D_D_to_n_He3']
            elif (reactant_1 == 'D' and reactant_2 == 'He3') or (reactant_1 == 'He3' and reactant_2 == 'D'):
                reactions = ['D_He3_to_p_alpha']
            elif reactant_1 == 'p' and reactant_2 == 'B11':
                reactions = ['p_B_to_3alpha']
            else:
                reactions = []

            for reaction in reactions:
                sigma_v_curr = get_sigma_v_fusion_sampled(Ti_keV, reaction=reaction)
                if reaction == 'p_B_to_3alpha':
                    # sigma_v_curr *= 1.3 # effetive increase of pB process reactivity
                    # sigma_v_curr *= 3.0  # effetive increase of pB process reactivity
                    pass
                if reactant_1 == reactant_2:
                    sigma_v_curr /= 2.0  # division by (1 + delta_jk)
                E_f_curr = get_E_reaction(reaction=reaction) * 1e3  # energy in [keV]
                E_ch_curr = get_E_charged(reaction=reaction) * 1e3  # energy in [keV]
                # E_ch_curr *= 0 # TODO: test
                denom_f_term += nj_list[ind_r_1] * nj_list[ind_r_2] * sigma_v_curr * (Q_fuel_inv * E_f_curr + E_ch_curr)
                P_fus_normalized += nj_list[ind_r_1] * nj_list[ind_r_2] * sigma_v_curr * E_f_curr * factor_keV_to_J

    f_T = 1.0  # Te / Ti
    nom_term = sum([nj * (1 + f_T * Zj) for nj, Zj in zip(nj_list, Zj_list)])

    # C_B = 0
    C_B = 5.34e-37  # brem constant [W m^3 keV^-0.5]
    C_B /= 10
    denom_brem_term = C_B * ne ** 2.0 * (f_T * Ti_keV) ** 0.5 * gamma_eff * factor_J_to_keV

    p_tau = 3.0 / 2 * nom_term ** 2.0 * Ti_keV ** 2.0 / (denom_f_term - denom_brem_term)
    p_tau[p_tau < 0] = np.nan

    ### Plot

    label = process
    try:
        ind_min = np.nanargmin(p_tau)
        label += ', $T_{min}$=' + '{:.1f}'.format(Ti_keV[ind_min]) + 'keV'
        label += ', $p\\tau$=' + '{:.1e}'.format(p_tau[ind_min])
        plt.figure(1)
        plt.plot(Ti_keV, p_tau, label=label, linewidth=2, color=color, linestyle=linestyle)
        plt.scatter(Ti_keV[ind_min], p_tau[ind_min], color=color)

        process_opt_dict[process] = {}
        process_opt_dict[process]['T_min'] = Ti_keV[ind_min]
        process_opt_dict[process]['p_tau_min'] = p_tau[ind_min]

        # calculate P_fus_normalized at opt paramters
        tau = 1.0  # [s]
        ni_normalization = (p_tau[ind_min] / tau) / (nom_term[ind_min] * Ti_keV[ind_min])
        # ni_normalization = 1e20
        # tau = p_tau[ind_min] / (nom_term[ind_min] * ni_normalization)
        process_opt_dict[process]['ni_opt'] = ni_normalization
        process_opt_dict[process]['tau'] = tau
        P_fus_opt = P_fus_normalized[ind_min] * ni_normalization ** 2.0 / 1e6  # [MW/m^3]
        process_opt_dict[process]['P_fus_opt'] = P_fus_opt

        ni_normalization = (p_tau / tau) / (nom_term * Ti_keV)
        P_fus = P_fus_normalized * ni_normalization ** 2.0 / 1e6  # [MW/m^3]
        process_opt_dict[process]['P_fus'] = P_fus

        ptaumin_rel = process_opt_dict[process]['p_tau_min'] / process_opt_dict['D-T']['p_tau_min']
        P_fus_rel = P_fus / ptaumin_rel ** 2.0
        process_opt_dict[process]['P_fus_rel'] = P_fus_rel

        plt.figure(2)
        plt.plot(Ti_keV, P_fus, label=process, linewidth=2, color=color, linestyle=linestyle)
        plt.scatter(Ti_keV[ind_min], P_fus[ind_min], color=color)

        plt.figure(3)
        plt.plot(Ti_keV, P_fus_rel, label=process, linewidth=2, color=color, linestyle=linestyle)
        plt.scatter(Ti_keV[ind_min], P_fus_rel[ind_min], color=color)

    except:
        print(process + ' failed')

# print final numbers and comparison between the processes
print('##########')
for process in process_opt_dict.keys():
    s = process
    ptaumin_rel = process_opt_dict[process]['p_tau_min'] / process_opt_dict['D-T']['p_tau_min']
    s += ', ptaumin_rel=' + '{:.1f}'.format(ptaumin_rel)
    s += ', ni_opt=' + '{:.1e}'.format(process_opt_dict[process]['ni_opt']) + ' m^-3'
    s += ', tau=' + '{:.1f}'.format(process_opt_dict[process]['tau']) + ' s'
    s += ', T_min=' + '{:.1f}'.format(process_opt_dict[process]['T_min']) + ' keV'
    s += ', P_fus_opt=' + '{:.1f}'.format(process_opt_dict[process]['P_fus_opt']) + ' MW/m^3'
    s += ', P_fus_opt_rel=' + '{:.2f}'.format(process_opt_dict[process]['P_fus_opt'] / ptaumin_rel ** 2.0) + ' MW/m^3'
    print(s)

## Plots

plt.figure(1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T_i$ [keV]')
# plt.ylabel('$n_i T_i \\tau_E$ [$m^{-3}$keV$\\cdot$s]')
# plt.ylabel('$p\\tau/k_B^2$ [$m^{-3}$keV$\\cdot$s]')
plt.ylabel('$p\\tau$ [$m^{-3}$keV$\\cdot$s]')
plt.title('Lawson $p \\tau$ for $Q=$' + str(Q))
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$P_{fus}$ [$MW/m^{-3}$]')
plt.title('Fusion power for Lawson $Q=$' + str(Q) + ' curve')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$P_{fus}$ [$MW/m^{-3}$]')
plt.title('Fusion power (relative to DT) for Lawson $Q=$' + str(Q) + ' curve')
plt.legend()
plt.grid(True)
plt.tight_layout()

# # ion fractions (interesting for catalyzed D-D process)
# plt.figure(3)
# for nj, ion in zip(nj_list, ions_list):
#     plt.plot(Ti_keV, nj, label=ion)
# plt.xscale('log')
# plt.xlabel('$T_i$ [keV]')
# plt.ylabel('ion fraction')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
