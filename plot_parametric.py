import matplotlib.pyplot as plt
import numpy as np

from default_settings import define_default_settings
from relaxation_algorithm_functions import load_simulation

# plt.close('all')

# parametric scan
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.01/'
save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.5_delta_n_factor_0.1/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.01_delta_n_factor_0.01/'
# save_dir_main = 'runs/runs_smooth_transition_adjust_cell_size_vth_right_bc_uniform_scaling_transition_n_factor_0.1_delta_n_factor_0.1_DT_mix/'


# Rm_list = np.array([1.4, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
# Rm_list = np.array([2.0, 2.5, 3.0])
# Rm_list = np.array([2.5, 3.0])
Rm_list = np.array([3.0])
U0_list = np.array([0, 1e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6])

for Rm in Rm_list:
    # print('Rm=' + str(Rm))
    flux_list = 0 * U0_list
    L_stable_list = 0 * U0_list
    n_stable_list = 0 * U0_list

    for i, U0 in enumerate(U0_list):
        # print('Rm=' + str(Rm) + ', U0=' + str(U0))

        save_dir = save_dir_main + '/Rm_' + str(Rm) + '_U_' + '{:.1e}'.format(U0)

        state_file = save_dir + '/state.pickle'
        settings_file = save_dir + '/settings.pickle'
        state, settings = load_simulation(state_file, settings_file)

        flux_list[i] = state['flux_mean']

        # if state['termination_criterion_reached'] is True:
        #     flux_list[i] = state['flux_mean']
        # else:
        #     print('failed: Rm=' + str(Rm) + ', U0=' + str(U0))
        #     flux_list[i] = np.nan

        # print(flux_list)

        ind_n_stable = np.where(state['n'] < settings['n_end'] * 1.01)[0][0]
        n_stable_list[i] = state['n'][ind_n_stable]
        z_array = np.cumsum(state['mirror_cell_sizes'])
        L_stable_list[i] = z_array[ind_n_stable]

    plt.figure(1)
    plt.plot(U0_list, flux_list, '-o', label='Rm=' + str(Rm))

    plt.figure(2)
    plt.plot(U0_list, L_stable_list, '-o', label='Rm=' + str(Rm))

    plt.figure(3)
    plt.plot(U0_list, n_stable_list, '-o', label='Rm=' + str(Rm))


plt.figure(1)
plt.xlabel('$U_0$ [m/s]')
plt.ylabel('flux')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(2)
plt.xlabel('$U_0$ [m/s]')
plt.ylabel('$L_{stable}$')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(3)
plt.xlabel('$U_0$ [m/s]')
plt.ylabel('$n_{stable}$')
plt.tight_layout()
plt.grid(True)
plt.legend()