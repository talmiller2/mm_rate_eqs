import matplotlib.pyplot as plt
import numpy as np

from default_settings import define_default_settings
from relaxation_algorithm_functions import load_simulation

# plt.close('all')

# parametric scan
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.5_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.01_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1_DT_mix/'

# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_1/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_1/'
# linestyle = '-'
# label_suffix = ' d=1'

# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_5e20/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_5e20/'
# linestyle = '--'
# label_suffix = ' d=3 $n_{rbc}=5e20$'

save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_3e21/'
# save_dir_main = 'runs/runs_smooth_transition_no_adaptive_mirror_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01_dim_3_rbc_3e21/'
linestyle = ':'
label_suffix = ' d=3 $n_{rbc}=3e21$'

# Rm_list = np.array([1.4, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
# Rm_list = np.array([2.0, 2.5, 3.0])
Rm_list = np.array([2.5, 3.0])
color_list = ['b', 'r']



# Rm_list = np.array([3.0])
# U0_list = np.array([0, 1e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
# U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4])

for ind_Rm, Rm in enumerate(Rm_list):
    color = color_list[ind_Rm]
    # print('Rm=' + str(Rm))
    flux_list = 0 * U0_list
    L_stable_list = 0 * U0_list
    n_stable_list = 0 * U0_list

    for ind_U0, U0 in enumerate(U0_list):
        # print('Rm=' + str(Rm) + ', U0=' + str(U0))

        # save_dir = save_dir_main + '/Rm_' + str(Rm) + '_U_' + '{:.1e}'.format(U0)
        save_dir = save_dir_main + '/Rm_' + str(Rm) + '_U_rel_' + str(U0)

        state_file = save_dir + '/state.pickle'
        settings_file = save_dir + '/settings.pickle'
        state, settings = load_simulation(state_file, settings_file)

        flux_list[ind_U0] = state['flux_mean']

        # if state['termination_criterion_reached'] is True:
        #     flux_list[i] = state['flux_mean']
        # else:
        #     print('failed: Rm=' + str(Rm) + ', U0=' + str(U0))
        #     flux_list[i] = np.nan

        # print(flux_list)

        ind_n_stable = np.where(state['n'] < settings['n_end'] * 1.01)[0][0]
        n_stable_list[ind_U0] = state['n'][ind_n_stable]
        z_array = np.cumsum(state['mirror_cell_sizes'])
        L_stable_list[ind_U0] = z_array[ind_n_stable]

        U0_to_plot = [0, 0.05, 0.1, 0.3]
        Rm_to_plot = 3.0
        z_array = np.cumsum(state['mirror_cell_sizes'])
        if U0 in U0_to_plot and Rm == Rm_to_plot:
            plt.figure(4 + ind_U0)
            plt.plot(z_array, state['n_tR'], linestyle=linestyle, color='b', label='n_tR' + label_suffix)
            plt.plot(z_array, state['n_tL'],  linestyle=linestyle, color='g', label='n_tL' + label_suffix)
            plt.plot(z_array, state['n_c'],  linestyle=linestyle, color='r', label='n_c'+ label_suffix)
            plt.plot(z_array, state['n'], linestyle=linestyle, color='k', label='n' + label_suffix)
            plt.title('$R_m$=' + str(Rm_to_plot) + ', $U/v_{th}$=' + str(U0))
            plt.ylabel('n $m^{-3}$')
            plt.xlabel('z [m]')
            plt.tight_layout()
            plt.grid(True)
            plt.legend()

    plt.figure(1)
    # plt.plot(U0_list, flux_list, '-o', label='Rm=' + str(Rm) + label_suffix, linestyle=linestyle, color=color)
    plt.plot(U0_list, flux_list/flux_list[0], '-o', label='Rm=' + str(Rm) + label_suffix, linestyle=linestyle, color=color)

    plt.figure(2)
    plt.plot(U0_list, L_stable_list, '-o', label='Rm=' + str(Rm) + label_suffix, linestyle=linestyle, color=color)

    plt.figure(3)
    plt.plot(U0_list, n_stable_list, '-o', label='Rm=' + str(Rm) + label_suffix, linestyle=linestyle, color=color)

plt.figure(1)
plt.xlabel('$U_0$ [m/s]')
# plt.ylabel('flux')
plt.ylabel('flux / flux0')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(2)
plt.xlabel('$U_0$ [m/s]')
plt.ylabel('$L_{stable}$ [m]')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(3)
plt.xlabel('$U_0$ [m/s]')
plt.ylabel('$n_{stable}$ [$m^{-3}$]')
plt.tight_layout()
plt.grid(True)
plt.legend()