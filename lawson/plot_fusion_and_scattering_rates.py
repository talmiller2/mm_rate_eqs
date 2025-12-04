import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['font.size'] = 14
plt.close('all')

from mm_rate_eqs.fusion_functions import get_sigma_v_fusion_sampled, get_reaction_label
from mm_rate_eqs.rate_functions import get_coulomb_scattering_rate
from mm_rate_eqs.default_settings import define_default_settings

# Ti_keV = np.linspace(1, 1000, 1000)
Ti_keV = np.logspace(0, 3, 3000)
Te_keV = 1.0 * Ti_keV

sigma_v_dict = {}
reactions = ['D_T_to_n_alpha', 'D_He3_to_p_alpha', 'D_D_to_p_T', 'D_D_to_n_He3', 'p_B_to_3alpha']
reaction_labels = [get_reaction_label(reaction) for reaction in reactions]

for reaction in reactions:
    sigma_v_dict[reaction] = get_sigma_v_fusion_sampled(Ti_keV, reaction=reaction)

ni = 1e21  # [m^-3]
settings = define_default_settings()
scat_rate_per_particle = get_coulomb_scattering_rate(ni, Ti_keV * 1e3, Te_keV * 1e3, settings, species='ions')
scat_rate = ni * scat_rate_per_particle

plt.figure(1, figsize=(8, 6))
colors = ['b', 'g', 'k', 'grey', 'r']
for reaction, reaction_label, color in zip(reactions, reaction_labels, colors):
    fusion_rate = ni ** 2 * sigma_v_dict[reaction]
    plt.plot(Ti_keV, fusion_rate, color=color, label=reaction_label, linewidth=2)
plt.plot(Ti_keV, scat_rate, color='k', linestyle='--', label='Coulomb scattering', linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('R [s$^{-1}$]')
plt.xlim([min(Ti_keV), max(Ti_keV)])
# plt.title('Fusion and Coulomb scattering rates')
plt.title('Fusion and Coulomb scattering rates, $n_i=10^{' + str(int(np.log10(ni))) + '}$[m$^{-3}$]')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ## save figs at higher res
# figs_folder = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/'
# plt.figure(1)
# plt.savefig(figs_folder + 'fusion_and_scattering_rates.pdf', format='pdf')
