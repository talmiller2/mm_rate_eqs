import matplotlib

# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import cm

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.fusion_functions import get_sigma_v_fusion, get_reaction_label, get_sigma_fusion, \
    get_sigma_v_fusion_approx
from mm_rate_eqs.rate_functions import get_thermal_velocity

### Plot fusion and radiation loss parameters

settings = define_default_settings()
Z_ion = 1
# B = 7 # [Tesla]
B = 15  # [Tesla]
# n0 = settings['n0']
n0 = 1e22  # = ni = ne
n_tot = 2 * n0
Ti_0 = settings['Ti_0']
Te_0 = settings['Te_0']

v_th = get_thermal_velocity(Ti_0, settings, species='ions')

# T_keV_array = np.linspace(0.2, 200, 1000)
T_keV_array = np.logspace(-1, 3, 1000)

reactions = []
reactions += ['D_T_to_n_alpha']
reactions += ['D_D_to_p_T_n_He3']
reactions += ['D_He3_to_p_alpha']
# reactions += ['T_T_to_alpha_2n']
reactions += ['p_B_to_3alpha']
# reactions += ['p_D_to_He3_gamma']
# reactions += ['He3_He3_to_alpha_2p']
reactions += ['p_p_to_D_e_nu']

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
    label = get_reaction_label(reaction=reaction) + ' (fit to exp data)'
    sigma_v_str = '$\\left< \\sigma \cdot v \\right>$'
    color = colors[ind_r]

    plt.figure(1)
    try:
        sigma_v_array = get_sigma_v_fusion(T_keV_array, reaction=reaction)
        plt.plot(T_keV_array, sigma_v_array, label=label, linewidth=3, color=color)
        # if reaction == 'p_B_to_3alpha':
        #     sigma_v_array = get_sigma_v_fusion(T_keV_array, reaction=reaction, use_resonance=False)
        #     label_res = get_reaction_label(reaction=reaction) + ' (fit lamda/o resonance term)'
        #     plt.plot(T_keV_array, sigma_v_array, label=label_res, linewidth=3, linestyle=':', color=color)
    except:
        pass

    sigma_v_approx_array = get_sigma_v_fusion_approx(T_keV_array, reaction=reaction)
    label_approx = get_reaction_label(reaction=reaction) + ' (Gamow approx)'
    # plt.plot(T_keV_array, sigma_v_approx_array, label=label_approx, linewidth=3, color=color, linestyle='--')
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
