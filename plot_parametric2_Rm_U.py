import matplotlib.pyplot as plt
import numpy as np

from default_settings import define_default_settings
from relaxation_algorithm_functions import load_simulation

# plt.close('all')

# parametric scan
save_dir_main = 'runs/small_experiment_adapative_mirror/'


Rm_list = np.array([2.0, 2.5, 3.0])
U0_list = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


for Rm in Rm_list:
    # print('Rm=' + str(Rm))
    flux_list = 0 * U0_list
    L_stable_list = 0 * U0_list
    n_stable_list = 0 * U0_list

    for i, U0 in enumerate(U0_list):
        # print('Rm=' + str(Rm) + ', U0=' + str(U0))

        # gas_name = 'hydrogen'
        # gas_name = 'helium'
        gas_name = 'lithium'
        # linestyle = '-'
        linestyle = '--'


        save_dir = save_dir_main + '/' + gas_name + '_Rm_' + str(Rm) + '_U_rel_' + str(U0)

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
    plt.plot(U0_list, flux_list, '-o', linestyle=linestyle, label=gas_name + ' Rm=' + str(Rm))

    plt.figure(2)
    plt.plot(U0_list, L_stable_list, '-o', linestyle=linestyle, label=gas_name + ' Rm=' + str(Rm))

    plt.figure(3)
    plt.plot(U0_list, n_stable_list, '-o', linestyle=linestyle, label=gas_name + ' Rm=' + str(Rm))


plt.figure(1)
plt.xlabel('$U_0/v_{th}$')
plt.ylabel('flux')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(2)
plt.xlabel('$U_0/v_{th}$')
plt.ylabel('$L_{stable}$')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(3)
plt.xlabel('$U_0/v_{th}$')
plt.ylabel('$n_{stable}$')
plt.tight_layout()
plt.grid(True)
plt.legend()