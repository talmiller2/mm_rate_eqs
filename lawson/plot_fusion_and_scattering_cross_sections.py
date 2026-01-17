import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 14
plt.close('all')

from mm_rate_eqs.fusion_functions import get_reaction_label, get_E_reaction, get_Zs_for_reaction, \
    get_sigma_fusion_sampled, get_sigma_v_fusion_sampled, get_sigma_fusion_approx, \
    get_sigma_v_fusion_approx_numeric_integration
from mm_rate_eqs.rate_functions import get_coulomb_scattering_cross_section

E_COM_keV = np.logspace(0, 4, 1000)

reactions = ['D_T_to_n_alpha']
reactions += ['D_D_to_p_T_n_He3']
reactions += ['D_He3_to_p_alpha']
reactions += ['p_B_to_3alpha']
# reactions += ['T_T_to_alpha_2n']
reaction_labels = [get_reaction_label(reaction) for reaction in reactions]

reactions_additional = []
# reactions_additional = ['p_p_to_D_e_nu', 'p_D_to_He3_gamma']
# reactions += reactions_additional
# reaction_labels += [get_reaction_label(reaction) for reaction in reactions_additional]

colors = ['b', 'g', 'k', 'r', 'orange', 'm']

plt.figure(1, figsize=(8, 6))
plt.figure(2, figsize=(8, 6))

for reaction, reaction_label, color in zip(reactions, reaction_labels, colors):

    # get fusion cross section
    if reaction not in reactions_additional:
        sigma_fusion = get_sigma_fusion_sampled(E_COM_keV, reaction=reaction)  # [barns]
    else:
        sigma_fusion = get_sigma_fusion_approx(E_COM_keV, reaction=reaction)  # [barns]

    # compare fusion and large-angle Coulomb scattering cross sections
    Z_1, Z_2 = get_Zs_for_reaction(reaction=reaction)
    sigma_scat = get_coulomb_scattering_cross_section(E_COM_keV, Z_1, Z_2)

    plt.figure(1)
    if reaction == reactions[0]:
        label = 'Coulomb scattering'
        linestyle_scat = 3
    else:
        label = None
        linestyle_scat = 2
    plt.plot(E_COM_keV, sigma_scat, color=color, label=label, linewidth=linestyle_scat, linestyle='--')
    plt.plot(E_COM_keV, sigma_fusion, color=color, label=reaction_label, linewidth=2, linestyle='-')
    # plt.plot(E_COM_keV, sigma_scat / sigma_fusion, color=color, label=reaction_label, linewidth=2, linestyle='-')

    # calculate colliding beam energy gain
    probability_fusion = sigma_fusion / (sigma_fusion + sigma_scat)
    E_out_MeV = probability_fusion * get_E_reaction(reaction)  # [MeV]
    E_out_keV = E_out_MeV * 1e3  # [keV]
    Q = E_out_keV / E_COM_keV

    plt.figure(2)
    plt.plot(E_COM_keV, Q, color=color, label=reaction_label, linewidth=2, linestyle='-')

plt.figure(1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$E_{COM}$ [keV]')
plt.ylabel('$\\sigma$ [barn]')
plt.xlim([min(E_COM_keV), max(E_COM_keV)])
plt.ylim([1e-4, 1e3])
plt.title('Fusion and Coulomb scattering cross sections')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(2)
plt.axhline(y=1, linestyle='--', color='grey', linewidth=3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$E_{COM}$ [keV]')
plt.ylabel('Q')
plt.xlim([min(E_COM_keV), max(E_COM_keV)])
plt.ylim([1e-4, 20])
plt.title('Colliding beam fusion gain $Q=E_{out}/E_{in}$')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ## save figs at higher res
# figs_folder = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/'
# plt.figure(1)
# plt.savefig(figs_folder + 'fusion_and_scattering_cross_sections.pdf', format='pdf')
# plt.figure(2)
# plt.savefig(figs_folder + 'fusion_colliding_beam_gain.pdf', format='pdf')
