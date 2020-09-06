import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation

plt.close('all')

main_dir = '../runs/slurm_runs/set2_Rm_3/'
# main_dir = '../runs/slurm_runs/set4_Rm_3_mfp_over_cell_4/'
# main_dir = '../runs/slurm_runs/set5_Rm_3_mfp_over_cell_20/'

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

# number_of_cells = 10
# number_of_cells = 20
number_of_cells = 30
# number_of_cells = 40
# number_of_cells = 50

U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

linestyles = []
linestyles += ['-']
linestyles += ['--']

LC_modes = []
LC_modes += ['sLC']
LC_modes += ['dLC']

for ind_mode in range(len(plasma_modes)):
    color = colors[ind_mode]
    plasma_mode = plasma_modes[ind_mode]

    for ind_LC in range(len(LC_modes)):
        linestyle = linestyles[ind_LC]
        LC_mode = LC_modes[ind_LC]

        flux_list = np.nan * np.zeros(len(U_list))
        for ind_U, U in enumerate(U_list):

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
                flux_list[ind_U] = state['flux_mean']

        # plot flux as a function of U
        # label_flux = plasma_modes[ind_mode] + '_N_' + str(number_of_cells) + '_' + LC_mode
        label_flux = plasma_modes[ind_mode] + '_' + LC_mode
        plt.figure(1)
        plt.plot(U_list, flux_list, '-', label=label_flux, linestyle=linestyle, color=color)
        plt.yscale("log")

plt.figure(1)
plt.xlabel('U / $v_{th}$')
plt.ylabel('flux')
# plt.title('flux as a function of MMM velocity')
plt.title('flux as a function of MMM velocity (N=' + str(number_of_cells) + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()
