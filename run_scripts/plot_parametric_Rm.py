import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation

# plt.close('all')

main_dir = '../runs/slurm_runs/set3_N_20/'

colors = []
colors += ['b']
colors += ['g']
colors += ['r']
colors += ['m']

plasma_modes = []
plasma_modes += ['isoTmfp']
# plasma_modes += ['isoT']
# plasma_modes += ['cool']
# plasma_modes += ['cool_mfpcutoff']

number_of_cells = 20
# U = 0
U = 0.1
# U = 0.2
Rm_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

linestyles = []
# linestyles += ['-']
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

        flux_list = np.nan * np.zeros(len(Rm_list))
        for ind_Rm, Rm in enumerate(Rm_list):

            run_name = plasma_mode
            run_name += '_Rm_' + str(Rm) + '_U_' + str(U)
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
                flux_list[ind_Rm] = state['flux_mean']

        # plot flux as a function of U
        # label_flux = plasma_modes[ind_mode] + '_N_' + str(number_of_cells) + '_' + LC_mode
        # label_flux = plasma_modes[ind_mode] + '_' + LC_mode
        label_flux = plasma_modes[ind_mode] + '_U_' + str(U) + '_' + LC_mode
        plt.figure(1)
        plt.plot(Rm_list, flux_list, '-', label=label_flux, linestyle=linestyle, color=color)
        plt.yscale("log")

plt.figure(1)
plt.xlabel('$R_m$')
plt.ylabel('flux')
# plt.title('flux as a function of mirror ratio (N=' + str(number_of_cells) + ', U/$v_{th}$=' + str(U) + ')')
plt.title('flux as a function of mirror ratio (N=' + str(number_of_cells) + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()
