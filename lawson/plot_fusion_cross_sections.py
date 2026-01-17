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
colors = ['b', 'g', 'k', 'r']
# colors = ['b', 'g', 'k', 'r', 'grey']
linestyles = ['-' for _ in range(len(reactions))]

reactions_additional = []
reactions_additional = ['p_p_to_D_e_nu', 'p_D_to_He3_gamma']
reactions += reactions_additional
reaction_labels += [get_reaction_label(reaction) for reaction in reactions_additional]
linestyles += ['--' for _ in range(len(reactions_additional))]
colors += ['orange', 'm']

plt.figure(1, figsize=(8, 6))

plot_reactivity_overlaid = False
# plot_reactivity_overlaid = True
if plot_reactivity_overlaid:
    plt.figure(2, figsize=(8, 6))

for reaction, reaction_label, color, linestyle in zip(reactions, reaction_labels, colors, linestyles):

    # get fusion cross section
    if reaction not in reactions_additional:
        sigma_fusion = get_sigma_fusion_sampled(E_COM_keV, reaction=reaction)  # [barns]
    else:
        sigma_fusion = get_sigma_fusion_approx(E_COM_keV, reaction=reaction)  # [barns]

    plt.figure(1)
    plt.plot(E_COM_keV, sigma_fusion, color=color, label=reaction_label, linewidth=2, linestyle=linestyle)

    if plot_reactivity_overlaid:
        # compare shape of mono-energetic sigma and Maxwell averaged
        if reaction not in reactions_additional:
            sigma_v_fusion = get_sigma_v_fusion_sampled(E_COM_keV, reaction=reaction)  # [barns]
        else:
            sigma_v_fusion = get_sigma_v_fusion_approx_numeric_integration(E_COM_keV, reaction=reaction)  # [barns]
        sigma_v_fac = np.nanmax(sigma_fusion) / np.nanmax(sigma_v_fusion)
        sigma_v_fusion *= sigma_v_fac
        Ti_equivalent_keV = 2 * E_COM_keV

        plt.figure(2)
        plt.plot(E_COM_keV, sigma_fusion, color=color, label=reaction_label, linewidth=2, linestyle=linestyle)
        plt.plot(Ti_equivalent_keV, sigma_v_fusion, color=color, label=None, linewidth=2, linestyle=':')

plt.figure(1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$E_{COM}$ [keV]')
plt.ylabel('$\\sigma$ [barn]')
plt.xlim([min(E_COM_keV), max(E_COM_keV)])
# sigma_min, sigma_max = 1e-4, 10
sigma_min, sigma_max = 1e-26, 10
plt.ylim([sigma_min, sigma_max])
plt.title('Fusion cross sections')
plt.legend()
plt.grid(True)
plt.tight_layout()

if plot_reactivity_overlaid:
    plt.figure(2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$E_{COM}$ [keV]')
    plt.ylabel('$\\sigma$ [barn]')
    plt.xlim([min(E_COM_keV), max(E_COM_keV)])
    plt.ylim([sigma_min, sigma_max])
    plt.title('Fusion cross sections (reactivity overlaid)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# ## save figs at higher res
# figs_folder = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/'
# plt.figure(1)
# # plt.savefig(figs_folder + 'fusion_cross_sections.pdf', format='pdf')
# plt.savefig(figs_folder + 'fusion_cross_sections_with_pp.pdf', format='pdf')
# # plt.figure(2)
# # plt.savefig(figs_folder + 'fusion_cross_sections_reactivity_overlaid.pdf', format='pdf')
