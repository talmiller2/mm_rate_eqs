import copy

import matplotlib.pyplot as plt
import numpy as np

# from mm_rate_eqs.loss_cone_functions import get_solid_angles, get_transverse_velocity_loss_cone_solutions
from mm_rate_eqs.loss_cone_functions import calc_axial_velocity_loss_cone_solutions, calc_special_U_values, \
    get_solid_angles


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

Rm_list = np.array([3])
colors = ['k']
Rm_list = np.array([2, 4, 10])
colors = ['b', 'g', 'r', 'orange']

do_plot_vlines = False
# do_plot_vlines = True

v = 1
U_list = np.linspace(0, 2.0 * v, 1000)

fig, axs = plt.subplots(1, 2, figsize=(16, 6), num=1)

for ind_Rm, Rm in enumerate(Rm_list):
    # U_list = np.linspace(0, 1.0 * v, 1000)

    vz_sol1, vz_sol2 = np.zeros(len(U_list)), np.zeros(len(U_list))
    omega_tR, omega_tL, omega_c = np.zeros(len(U_list)), np.zeros(len(U_list)), np.zeros(len(U_list))

    for ind_U, U in enumerate(U_list):
        # calculate the transverse velocity solutions
        # v_perp_high[ind_U], v_perp_low[ind_U], U_transition, U_last_sol = get_transverse_velocity_loss_cone_solutions(U, Rm)

        vz_sol1[ind_U], vz_sol2[ind_U] = calc_axial_velocity_loss_cone_solutions(U, v, Rm)
        U_transition, U_last_sol = calc_special_U_values(v, Rm)

        # calculate the solid angles for right/left loss cones
        omega_tR[ind_U], omega_tL[ind_U], omega_c[ind_U] = get_solid_angles(U, v, 1 / Rm)


    def plot_vlines(ax):
        ax.axvline(1 / v,
                   # linestyle='-',
                   linewidth=2,
                   # alpha=0.5,
                   color='b',
                   # color=colors[ind_Rm],
                   label='$U=v$')
        ax.axvline(U_transition / v,
                   # linestyle='--',
                   linewidth=2,
                   # alpha=0.5,
                   color='g',
                   # color=colors[ind_Rm],
                   label='$U_{trans}$')
        ax.axvline(U_last_sol / v,
                   # linestyle=':',
                   linewidth=2,
                   # alpha=0.5,
                   color='r',
                   # color=colors[ind_Rm],
                   label='$U_{last}$')


    axs[0].plot(U_list / v, (vz_sol1 + U_list) / v,
                label='$v_{z,1}$ ($R_m$=' + str(Rm) + ')',
                color=colors[ind_Rm])
    axs[0].plot(U_list / v, (vz_sol2 + U_list) / v, linestyle='--',
                label='$v_{z,2}$',
                color=colors[ind_Rm])
    if do_plot_vlines:
        plot_vlines(axs[0])

    axs[1].plot(U_list / v, omega_tR,
                label='$\\alpha_r$ ($R_m$=' + str(Rm) + ')',
                color=colors[ind_Rm])
    axs[1].plot(U_list / v, omega_tL, '--',
                label='$\\alpha_l$',
                color=colors[ind_Rm])
    axs[1].plot(U_list / v, omega_c, ':',
                label='$\\alpha_c$',
                color=colors[ind_Rm])
    if do_plot_vlines:
        plot_vlines(axs[1])

axs[0].set_xlim([0, max(U_list) / v])
axs[0].set_xlabel('$U/v$')
axs[0].set_ylabel('$(v_{z} + U)/v$ solutions')
axs[0].legend(loc='upper left')
axs[0].grid()

axs[1].set_xlim([0, max(U_list) / v])
axs[1].set_xlabel('$U/v$')
axs[1].set_ylabel('$\\alpha=\\Omega$ /4$\\pi$ normalized solid angles')
axs[1].legend(loc='upper left')
axs[1].grid()

fig.tight_layout()

# ### saving figures
# fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2026/pics/'
# # file_name = 'MMM_loss_cones_single_Rm'
# file_name = 'MMM_loss_cones_multiple_Rm'
# fig.savefig(fig_save_dir + file_name + '.pdf', format='pdf', dpi=600)
