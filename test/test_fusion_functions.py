import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.fusion_functions import get_sigma_v_fusion, get_E_reaction, get_brem_radiation_loss, \
    get_cyclotron_radiation_loss, get_Z_ion_for_reaction, \
    get_fusion_power, get_lawson_parameters, get_reaction_label
from mm_rate_eqs.rate_functions import get_thermal_velocity

### Plot fusion and radiation loss parameters

settings = define_default_settings()
keV = settings['keV']
kB_eV = settings['kB_eV']
eV_to_K = settings['eV_to_K']
# Z_ion = settings['Z_ion']
Z_ion = 1
# B = 7 # [Tesla]
B = 15  # [Tesla]
# n0 = settings['n0']
n0 = 1e22  # = ni = ne
n_tot = 2 * n0
Ti_0 = settings['Ti_0']
Te_0 = settings['Te_0']

v_th = get_thermal_velocity(Ti_0, settings, species='ions')

T_keV_array = np.linspace(0.2, 200, 1000)
T_eV_array = T_keV_array * 1e3
# reactions = ['D_T_to_n_alpha', 'D_D_to_p_T', 'D_D_to_n_He3', 'D_He3_to_p_alpha']
reactions = ['D_T_to_n_alpha', 'D_D_to_p_T_n_He3', 'D_He3_to_p_alpha', 'p_B_to_3alpha']
# reactions = ['D_T_to_n_alpha']
# reactions = ['D_D_to_p_T', 'D_D_to_n_He3', 'D_D_to_p_T_n_He3']
# reactions = ['D_T_to_n_alpha', 'p_B_to_3alpha', 'p_B_to_3alpha_v2']
# reactions = ['p_B_to_3alpha']

colors = ['b', 'g', 'r', 'k', 'y', 'c']

plt.rcParams.update({'font.size': 16})
plt.close('all')

plt.figure()
# for reaction in reactions:
for ind_r, reaction in enumerate(reactions):
    E_reaction = get_E_reaction(reaction=reaction)
    # Z_ion = get_Z_ion_for_reaction(reaction=reaction)
    label = get_reaction_label(reaction=reaction) + ' ($E$=' + str(round(E_reaction, 3)) + 'MeV)'
    sigma_v_str = '$\\left< \\sigma \cdot v \\right>$'
    color = colors[ind_r]

    plt.figure(1)
    sigma_v_array = get_sigma_v_fusion(T_eV_array, reaction=reaction)
    plt.plot(T_keV_array, sigma_v_array, label=label, linewidth=3, color=color)
    if reaction == 'p_B_to_3alpha':
        sigma_v_array = get_sigma_v_fusion(T_eV_array, reaction=reaction, use_resonance=False)
        label_res = label + ' w/o res'
        plt.plot(T_keV_array, sigma_v_array, label=label_res, linewidth=3, linestyle='--', color=color)
    plt.legend()
    plt.ylabel(sigma_v_str + ' $[m^3/s]$')
    plt.xlabel('T [keV]')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)

    plt.figure(2)
    sigma_v_array = get_sigma_v_fusion(T_keV_array * 1e3, reaction=reaction)
    funsion_scaling_power = np.diff(np.log(sigma_v_array)) / np.diff(np.log(T_keV_array))
    plt.plot(T_keV_array[1:], funsion_scaling_power, label=label, linewidth=3, color=color)
    plt.legend()
    plt.title(sigma_v_str + ' temperature scaling power $\propto T^{\\nu}$')
    plt.ylabel('$\\nu$ [dimensionless]')
    plt.xlabel('T [keV]')
    plt.xscale('log')
    # plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)

    plt.figure(3)
    if ind_r == 0:
        sigma_v_array_ref = get_sigma_v_fusion(T_eV_array, reaction=reaction)
    sigma_v_array = get_sigma_v_fusion(T_eV_array, reaction=reaction)
    plt.plot(T_keV_array, sigma_v_array / sigma_v_array_ref, label=label, linewidth=3, color=color)
    plt.legend()
    plt.title(sigma_v_str + ' relative to ' + get_reaction_label(reaction=reactions[0]))
    plt.xlabel('T [keV]')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)

    # Radiation and Fusion, assuming Ti=Te
    P_brem_radiation_loss_volumetric = get_brem_radiation_loss(n0, n0, T_keV_array, Z_ion)  # W/m^3
    P_cycl_radiation_loss_volumetric = get_cyclotron_radiation_loss(n0, T_keV_array, B)  # W/m^3
    P_cycl_radiation_loss_volumetric_total = P_brem_radiation_loss_volumetric + P_cycl_radiation_loss_volumetric
    plt.figure(4)
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric, label=label, linewidth=3, color=color)
    if ind_r == len(reactions) - 1:
        label_brem = 'Brem loss, Z=' + str(Z_ion) + ', n=' + str(n_tot)
        plt.plot(T_keV_array, P_brem_radiation_loss_volumetric, '--', label=label_brem, linewidth=3, color='c')
        label_cyc = 'Cyclotron loss, B=' + str(B) + 'T' + ', n=' + str(n_tot)
        plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric, ':', label=label_cyc, linewidth=3, color='c')
        plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric_total, '-', label='Total radiation loss', linewidth=3,
                 color='c')
    plt.legend()
    plt.title('Fusion and radiation loss power, $T_i=T_e$')
    plt.xlabel('$T_i$ [keV]')
    plt.ylabel('$W/m^3$')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)

    plt.figure(5)
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric / P_cycl_radiation_loss_volumetric_total, label=label, linewidth=3,
             color=color)
    plt.legend()
    plt.title('Fusion to radiation loss ratio, $T_i=T_e$')
    plt.xlabel('$T_i$ [keV]')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)

# Summary of Lawson criterion
print('#### Summary of Lawson criterion ###')

tau_lawson, flux_lawson = get_lawson_parameters(n0, Ti_0, settings)
print('tau_lawson: ', '{:.3e}'.format(tau_lawson), 's')
print('flux_lawson: ', '{:.3e}'.format(flux_lawson), 's^-1')
v_th = get_thermal_velocity(Ti_0, settings)
flux_single_mirror = v_th * n0 * settings['cross_section_main_cell']
print('flux single mirror: ', '{:.3e}'.format(flux_single_mirror), 's^-1')

# Fusion power in nominal parameters
print('Main cell volume: ', '{:.3e}'.format(settings['volume_main_cell']), 'm^3')
P_fusion_volumetric = get_fusion_power(n0, Ti_0 / settings['keV'])
P_fusion = P_fusion_volumetric * settings['volume_main_cell']  # Watt
print('Fusion power: ', '{:.3e}'.format(P_fusion / 1e6), 'MW')
