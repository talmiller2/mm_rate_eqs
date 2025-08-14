import copy

import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.loss_cone_functions import get_solid_angles

plt.close('all')

axes_label_size = 12
# axes_label_size = 14
# axes_label_size = 18
title_fontsize = 14
legend_fontsize = 12

Rm_list = np.array([2, 3, 10])
# Rm_list = np.array([1.2, 2, 3, 10])
colors = ['b', 'g', 'r', 'orange']

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for ind_Rm, Rm in enumerate(Rm_list):
    alpha = 1 / Rm
    vth = 1
    U_list = np.linspace(0, 2.0 * vth, 1000)
    # U_list = np.linspace(0, 1.0 * vth, 1000)
    omega_tR, omega_tL, omega_c = np.zeros(len(U_list)), np.zeros(len(U_list)), np.zeros(len(U_list))
    for ind_U, U in enumerate(U_list):
        # calculate the solid angles for right/left loss cones
        omega_tR[ind_U], omega_tL[ind_U], omega_c[ind_U] = get_solid_angles(U, vth, alpha)

    ax.plot(U_list / vth, omega_tR,
            # label='$\\Omega_{tR}$ Rm=' + str(Rm),
            label='$\\alpha_r$ $R_m$=' + str(Rm),
            color=colors[ind_Rm])
    ax.plot(U_list / vth, omega_tL, '--',
            # label='$\\Omega_{tL}$ Rm=' + str(Rm),
            label='$\\alpha_l$',
            color=colors[ind_Rm])
    plt.plot(U_list / vth, omega_c, ':',
             label='$\\alpha_c$',
             color=colors[ind_Rm])

ax.set_xlabel('$U/v_{th}$', fontsize=axes_label_size)
# ax.set_ylabel('$\\Omega$ /4$\\pi$', fontsize=axes_label_size)
ax.set_ylabel('$\\alpha=\\Omega$ /4$\\pi$', fontsize=axes_label_size)
ax.set_title('MMM modified loss cone solid angles', fontsize=title_fontsize)
ax.legend(fontsize=legend_fontsize)
ax.grid()
fig.tight_layout()

# ### saving figures
# fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2026/pics/'
# file_name = 'MMM_loss_cones_solid_angles'
# fig.savefig(fig_save_dir + file_name + '.pdf', format='pdf', dpi=600)
