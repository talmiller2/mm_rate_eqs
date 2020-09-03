import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation

plt.close('all')

main_dir = '../runs/slurm_runs/set2_Rm_3/'

colors = []
colors += ['b']
colors += ['g']
colors += ['r']
colors += ['m']

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
plasma_modes += ['cool']
plasma_modes += ['cool_mfpcutoff']

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# num_cells_list = [3, 5, 8]

U = 0
# U = 0.05
# U = 0.1
# U = 0.3
# U = 0.5

linestyles = []
linestyles += ['-']
linestyles += ['--']

LC_modes = []
LC_modes += ['sLC']
# LC_modes += ['dLC']


for ind_mode in range(len(plasma_modes)):
    color = colors[ind_mode]
    plasma_mode = plasma_modes[ind_mode]

    for ind_LC in range(len(LC_modes)):
        linestyle = linestyles[ind_LC]
        LC_mode = LC_modes[ind_LC]

        flux_list = np.nan * np.zeros(len(num_cells_list))
        for ind_N, number_of_cells in enumerate(num_cells_list):

            run_name = plasma_mode
            run_name += '_N_' + str(number_of_cells) + '_U_' + str(U)
            run_name += '_' + LC_mode
            # print('run_name = ' + run_name)

            save_dir = main_dir + '/' + run_name

            state_file = save_dir + '/state.pickle'
            settings_file = save_dir + '/settings.pickle'
            state, settings = load_simulation(state_file, settings_file)

            # flux_list[ind_N] = state['flux_mean']
            # if state['successful_termination'] == False:
            #     print('RUN FAILED.')

            if state['successful_termination'] == True:
                flux_list[ind_N] = state['flux_mean']

            # extract the density profile
            chosen_num_cells = 50
            if number_of_cells == chosen_num_cells:
                plt.figure(2)
                label = run_name
                plt.plot(state['n'], '-', label=label, linestyle=linestyle, color=color)

        # plot flux as a function of N
        label_flux = plasma_modes[ind_mode] + '_U_' + str(U) + '_' + LC_mode
        plt.figure(1)
        plt.plot(num_cells_list, flux_list, '-', label=label_flux, linestyle=linestyle, color=color)
        plt.yscale("log")

plt.figure(1)
plt.xlabel('number of cells')
plt.ylabel('flux')
plt.title('flux as a function of system size')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(2)
plt.xlabel('cell number')
plt.ylabel('density')
plt.title('density profile for N = ' + str(chosen_num_cells) + ' cells')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.grid(True)
