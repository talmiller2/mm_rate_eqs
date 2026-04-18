import copy

import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.loss_cone_functions import get_solid_angles, get_transverse_velocity_loss_cone_solutions

plt.close('all')

plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 3

# # axes_label_size = 12
# # axes_label_size = 14
# axes_label_size = 16
# title_fontsize = 14
# # legend_fontsize = 12
# # legend_fontsize = 14
# legend_fontsize = 16

Rm_list = np.array([2, 3, 10])
# Rm_list = np.array([1.2, 2, 3, 10])
colors = ['b', 'g', 'r', 'orange']

vth = 1
U_list = np.linspace(0, 2.0 * vth, 1000)

##### solid angle plots #####

fig, ax = plt.subplots(1, 1, figsize=(8, 6), num=1)
for ind_Rm, Rm in enumerate(Rm_list):
    alpha = 1 / Rm
    omega_tR, omega_tL, omega_c = np.zeros(len(U_list)), np.zeros(len(U_list)), np.zeros(len(U_list))
    for ind_U, U in enumerate(U_list):
        # calculate the solid angles for right/left loss cones
        omega_tR[ind_U], omega_tL[ind_U], omega_c[ind_U] = get_solid_angles(U, vth, alpha)

    ax.plot(U_list / vth, omega_tR,
            # label='$\\Omega_{tR}$ Rm=' + str(Rm),
            label='$\\alpha_r$ ($R_m$=' + str(Rm) + ')',
            color=colors[ind_Rm])
    ax.plot(U_list / vth, omega_tL, '--',
            # label='$\\Omega_{tL}$ Rm=' + str(Rm),
            label='$\\alpha_l$',
            color=colors[ind_Rm])
    plt.plot(U_list / vth, omega_c, ':',
             label='$\\alpha_c$',
             color=colors[ind_Rm])

ax.set_xlim([0, max(U_list) / vth])
ax.set_xlabel('$U/v_{th}$')
ax.set_ylabel('$\\alpha=\\Omega$ /4$\\pi$ normalized solid angles')
# ax.set_title('MMM modified loss cone solid angles')
ax.legend(loc='right')
ax.grid()
fig.tight_layout()

# ### saving figures
# fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2026/pics/'
# file_name = 'MMM_loss_cones_solid_angles'
# fig.savefig(fig_save_dir + file_name + '.pdf', format='pdf', dpi=600)
