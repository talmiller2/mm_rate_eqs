import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.fusion_functions import get_sigma_v_fusion, get_fusion_power, get_lawson_parameters, \
    get_reaction_label, get_sigma_fusion, get_sigma_v_fusion_approx
from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss
from mm_rate_eqs.rate_functions import get_thermal_velocity

### Plot fusion and radiation loss parameters

settings = define_default_settings()

# Z_ion = settings['Z_ion']
Z_ion = 1
# B = 7 # [Tesla]
B = 15  # [Tesla]
# n0 = settings['n0']
n0 = 2e22  # = ni = ne
n_tot = 2 * n0
Ti_0 = settings['Ti_0']
Te_0 = settings['Te_0']

v_th = get_thermal_velocity(Ti_0, settings, species='ions')

T_keV_array = np.linspace(0.2, 200, 1000)

reactions = []
reactions += ['D_T_to_n_alpha']
reactions += ['D_D_to_p_T_n_He3']
reactions += ['D_He3_to_p_alpha']
# reactions += ['T_T_to_alpha_2n']
reactions += ['p_B_to_3alpha']
# reactions += ['p_D_to_He3_gamma']
# reactions += ['He3_He3_to_alpha_2p']
# reactions += ['p_p_to_D_e_nu']

# colors = ['b', 'g', 'r', 'k', 'm', 'y', 'c', 'b']
colors = cm.rainbow(np.linspace(0, 1, len(reactions)))

# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 14})
plt.close('all')

plt.figure()
# for reaction in reactions:
for ind_r, reaction in enumerate(reactions):
    # E_reaction = get_E_reaction(reaction=reaction)
    # label = get_reaction_label(reaction=reaction) + ' ($E$=' + str(round(E_reaction, 3)) + 'MeV)'
    label = get_reaction_label(reaction=reaction)
    label_sigmav = get_reaction_label(reaction=reaction) + ' (fit to exp data)'
    sigma_v_str = '$\\left< \\sigma \cdot v \\right>$'
    color = colors[ind_r]

    plt.figure(1)
    try:
        sigma_v_array = get_sigma_v_fusion(T_keV_array, reaction=reaction)
        plt.plot(T_keV_array, sigma_v_array, label=label_sigmav, linewidth=3, color=color)
        # if reaction == 'p_B_to_3alpha':
        #     sigma_v_array = get_sigma_v_fusion(T_keV_array, reaction=reaction, use_resonance=False)
        #     label_res = get_reaction_label(reaction=reaction) + ' (fit w/o resonance term)'
        #     plt.plot(T_keV_array, sigma_v_array, label=label_res, linewidth=3, linestyle=':', color=color)
    except:
        pass

    sigma_v_approx_array = get_sigma_v_fusion_approx(T_keV_array, reaction=reaction)
    label_approx = get_reaction_label(reaction=reaction) + ' (Gamow approx)'
    plt.plot(T_keV_array, sigma_v_approx_array, label=label_approx, linewidth=3, color=color, linestyle='--')
    # label_approx = get_reaction_label(reaction=reaction)
    # plt.plot(T_keV_array, sigma_v_approx_array, label=label_approx, linewidth=3, color=color, linestyle='-')

    # sigma_v_approx_array = get_sigma_v_fusion_approx(T_keV_array, reaction=reaction, n=n_tot)
    # label_approx = get_reaction_label(reaction=reaction) + ' (Gamow approx + screening)'
    # plt.plot(T_keV_array, sigma_v_approx_array, label=label_approx, linewidth=3, color=color, linestyle=':')

    plt.legend()
    plt.ylabel(sigma_v_str + ' $[m^3/s]$')
    plt.xlabel('T [keV]')
    plt.title('Maxwell-averaged fusion reactivity')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)

    plt.figure(2)
    E_array = T_keV_array
    sigma_array = get_sigma_fusion(T_keV_array, reaction=reaction)
    label_crs = get_reaction_label(reaction=reaction)
    plt.plot(E_array, sigma_array, label=label_crs, linewidth=3, color=color)
    plt.legend()
    plt.ylabel('$\\sigma$ [barn]')
    plt.xlabel('$E_{COM}$ [keV]')
    plt.title('Fusion cross section (Gamow approx)')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)

    plt.figure(3)
    sigma_v_array = get_sigma_v_fusion(T_keV_array, reaction=reaction)
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

    plt.figure(4)
    if ind_r == 0:
        sigma_v_array_ref = get_sigma_v_fusion(T_keV_array, reaction=reaction)
    sigma_v_array = get_sigma_v_fusion(T_keV_array, reaction=reaction)
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
    plt.figure(5)
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

    plt.figure(6)
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
