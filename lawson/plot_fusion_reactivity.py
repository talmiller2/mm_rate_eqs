import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['font.size'] = 12
plt.close('all')

from mm_rate_eqs.fusion_functions import get_sigma_v_fusion_sampled, get_reaction_label, get_E_reaction

# Ti_keV = np.linspace(1, 1000, 1000)
Ti_keV = np.logspace(0, 3, 3000)

sigma_v_dict = {}
reactions = ['D_T_to_n_alpha', 'D_He3_to_p_alpha', 'D_D_to_p_T', 'D_D_to_n_He3', 'p_B_to_3alpha']
reaction_labels = [get_reaction_label(reaction) for reaction in reactions]

for reaction in reactions:
    sigma_v_dict[reaction] = get_sigma_v_fusion_sampled(Ti_keV, reaction=reaction)

# # load the data from Sikora-Weller-2016
# data_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Books/Fusion cross sections/2016-Sikora-Weller-cross-section/'
# reactions2 = ['p_B_to_3alpha Sikora(2016) old', 'p_B_to_3alpha Sikora(2016) new']
# reactions_labels2 = [reaction_labels[-1] + ' Sikora(2016) old', reaction_labels[-1] + ' Sikora(2016) new']
# file_names = ['old_data', 'new_data']
# for reaction, reaction_label, file_name in zip(reactions2, reactions_labels2, file_names):
#     data_file = data_dir + '/' + file_name + '.csv'
#     data = np.loadtxt(data_file, delimiter=',', skiprows=1)
#     T_data = data[:, 0]
#     sigma_v_data = data[:, 1] * 1e-6  # change units [cm^3/s] to [m^3/s]
#     interp_fun = interp1d(T_data, sigma_v_data, bounds_error=False)
#     # sigma_v_interped = 10.0 ** (interp_fun(np.log10(Ti_keV)))
#     sigma_v_interped = interp_fun(Ti_keV)
#     sigma_v_dict[reaction] = sigma_v_interped

plt.figure(1, figsize=(7, 5))
colors = ['b', 'g', 'k', 'grey', 'r']
for reaction, reaction_label, color in zip(reactions, reaction_labels, colors):
    plt.plot(Ti_keV, sigma_v_dict[reaction], color=color, label=reaction_label, linewidth=2)
# for reaction, reaction_label, color in zip(reactions2, reactions_labels2, ['r', 'orange']):
#     plt.plot(Ti_keV, sigma_v_dict[reaction], color=color, label=reaction_label, linewidth=2, linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$\\left< \\sigma \cdot v \\right>$ [m$^3$/s]')
plt.xlim([min(Ti_keV), max(Ti_keV)])
plt.title('Fusion reactivity')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(2, figsize=(7, 5))
for reaction, reaction_label, color in zip(reactions, reaction_labels, colors):
    E_fus = get_E_reaction(reaction)  # [MeV]
    plt.plot(Ti_keV, E_fus * sigma_v_dict[reaction], color=color, label=reaction_label, linewidth=2)
# for reaction, reaction_label, color in zip(reactions2, reactions_labels2, ['r', 'orange']):
#     plt.plot(Ti_keV, sigma_v_dict[reaction], color=color, label=reaction_label, linewidth=2, linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$\\left< \\sigma \cdot v \\right>E_{fus}$ [m$^3$MeV/s]')
plt.xlim([min(Ti_keV), max(Ti_keV)])
plt.title('Fusion reactivity $\\times$ fusion energy per reaction')
plt.legend()
plt.grid(True)
plt.tight_layout()

## save figs at higher res
figs_folder = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/lawson_plots/'
plt.figure(1)
plt.savefig(figs_folder + 'fusion_reactivities.pdf', format='pdf')
plt.figure(2)
plt.savefig(figs_folder + 'fusion_reactivities_times_energy.pdf', format='pdf')
