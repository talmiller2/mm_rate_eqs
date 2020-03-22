import matplotlib.pyplot as plt
import numpy as np

from default_settings import define_default_settings


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
        print('invalid reaction:', reaction)
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
        print('invalid reaction:', reaction)
    return E_charged


def get_fusion_sigma_v_E_reaction(T_eV, reaction='D-T_to_n_alpha'):
    if reaction == 'D-D_to_p_T_n_He3':
        sigma_v_E = get_sigma_v_fusion(T_eV, reaction='D-D_to_p_T') * get_E_reaction(reaction='D-D_to_p_T') \
                    + get_sigma_v_fusion(T_eV, reaction='D-D_to_n_He3') * get_E_reaction(reaction='D-D_to_n_He3')
    else:
        sigma_v_E = get_sigma_v_fusion(T_eV, reaction=reaction) * get_E_reaction(reaction=reaction)
    return sigma_v_E * 1e6 * 1.6e-19  # MeV to J


def get_fusion_power(n, T_keV, reaction='D-T_to_n_alpha'):
    return 0.25 * n ** 2 * get_fusion_sigma_v_E_reaction(T_keV * 1e3, reaction=reaction)


def get_brem_radiation_loss(ni, ne, Te, Z_charge):
    # Fusion Plasma Analysis, p. 228, Bremsstrahlung radiation [W/m^3], T in [keV]
    return 4.8e-37 * Z_charge ** 2 * ni * ne * Te ** (0.5)


def get_cyclotron_radiation_loss(ne, Te, B):
    # Fusion Plasma Analysis, p. 231, cyclotron radiation [W/m^3], B in [Tesla], T in [keV]
    return 6.2e-17 * B ** 2 * ne * Te


### Plot fusion and radiation loss parameters

settings = define_default_settings()
keV = settings['keV']
eV_to_K = settings['eV_to_K']
Z_charge = settings['Z_charge']
B = settings['B']
n0 = settings['n0']
Ti_0 = settings['Ti_0']
Te_0 = settings['Te_0']

T_keV_array = np.linspace(0.2, 200, 1000)
# reactions = ['D-T_to_n_alpha', 'D-D_to_p_T', 'D-D_to_n_He3', 'He3-D_to_p_alpha']
# reactions = ['D-T_to_n_alpha', 'D-D_to_p_T_n_He3', 'He3-D_to_p_alpha', 'p_B_to_3alpha']
reactions = ['D-T_to_n_alpha', 'D-D_to_p_T_n_He3']

plt.rcParams.update({'font.size': 16})
plt.close('all')

plt.figure()
for reaction in reactions:
    sigma_v_array = get_sigma_v_fusion(T_keV_array * 1e3, reaction=reaction)
    E_reaction = get_E_reaction(reaction=reaction)
    label = reaction + ', $E_{reaction}$=' + str(round(E_reaction, 3)) + 'MeV'
    plt.plot(T_keV_array, sigma_v_array, label=label, linewidth=3)
plt.legend()
plt.ylabel('$\\sigma*v [m^3/s]$')
plt.xlabel('T [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

plt.figure()
sigma_v_array_ref = get_sigma_v_fusion(T_keV_array * 1e3, reaction=reactions[0])
for reaction in reactions:
    sigma_v_array = get_sigma_v_fusion(T_keV_array * 1e3, reaction=reaction)
    E_reaction = get_E_reaction(reaction=reaction)
    label = reaction + ', $E_{reaction}$=' + str(round(E_reaction, 3)) + 'MeV'
    plt.plot(T_keV_array, sigma_v_array / sigma_v_array_ref, label=label, linewidth=3)
plt.legend()
plt.ylabel('$\\sigma*v$ relative to DT')
plt.xlabel('T [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

# Radiation and Fusion, assuming Ti=Te
P_brem_radiation_loss_volumetric = get_brem_radiation_loss(n0, n0, T_keV_array, Z_charge)  # W/m^3
P_cycl_radiation_loss_volumetric = get_cyclotron_radiation_loss(n0, T_keV_array, B)  # W/m^3
P_cycl_radiation_loss_volumetric_total = P_brem_radiation_loss_volumetric + P_cycl_radiation_loss_volumetric
plt.figure()
plt.plot(T_keV_array, P_brem_radiation_loss_volumetric, '--', label='brem loss', linewidth=3)
plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric, '--', label='cyclotron loss', linewidth=3)
plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric_total, '--', label='total loss', linewidth=3)
for reaction in reactions:
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric, label=reaction, linewidth=3)
plt.legend()
plt.title('Fusion and radiation loss power, $T_i=T_e$')
plt.xlabel('$T_i$ [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

plt.figure()
for reaction in reactions:
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric / P_cycl_radiation_loss_volumetric_total, label=reaction, linewidth=3)
plt.legend()
plt.title('Fusion to radiation loss ratio, $T_i=T_e$')
plt.xlabel('$T_i$ [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

# Radiation and Fusion, assuming Ti=3*Te
P_brem_radiation_loss_volumetric = get_brem_radiation_loss(n0, n0, T_keV_array/3.0, Z_charge)  # W/m^3
P_cycl_radiation_loss_volumetric = get_cyclotron_radiation_loss(n0, T_keV_array/3.0, B)  # W/m^3
P_cycl_radiation_loss_volumetric_total = P_brem_radiation_loss_volumetric + P_cycl_radiation_loss_volumetric
plt.figure()
plt.plot(T_keV_array, P_brem_radiation_loss_volumetric, '--', label='brem loss', linewidth=3)
plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric, '--', label='cyclotron loss', linewidth=3)
plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric_total, '--', label='total loss', linewidth=3)
for reaction in reactions:
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric, label=reaction, linewidth=3)
plt.legend()
plt.title('Fusion and radiation loss power, $T_i=3T_e$')
plt.xlabel('$T_i$ [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

plt.figure()
for reaction in reactions:
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric / P_cycl_radiation_loss_volumetric_total, label=reaction, linewidth=3)
plt.legend()
plt.title('Fusion to radiation loss ratio, $T_i=3T_e$')
plt.xlabel('$T_i$ [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()