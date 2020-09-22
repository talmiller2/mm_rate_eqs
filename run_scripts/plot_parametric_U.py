import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 14})

import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation


def define_plasma_mode_label(plasma_mode):
    label = ''
    if plasma_mode == 'isoT':
        label += 'isothermal'
    elif plasma_mode == 'isoTmfp':
        label += 'isothermal iso-mfp'
    elif 'cool' in plasma_mode:
        plasma_dimension = int(plasma_mode.split('d')[-1])
        label += 'cooling d=' + str(plasma_dimension)
    return label


def define_LC_mode_label(LC_mode):
    label = ''
    if LC_mode == 'sLC':
        label += 'with static-LC'
    else:
        label += 'with dynamic-LC'
    return label


def define_label(plasma_mode, LC_mode):
    label = define_plasma_mode_label(plasma_mode)
    label += ', '
    label += define_LC_mode_label(LC_mode)
    return label


plt.close('all')

# main_dir = '../runs/slurm_runs/set2_Rm_3/'
# main_dir = '../runs/slurm_runs/set4_Rm_3_mfp_over_cell_4/'
# main_dir = '../runs/slurm_runs/set5_Rm_3_mfp_over_cell_20/'
main_dir = '../runs/slurm_runs/set6_Rm_3_mfp_over_cell_1_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set7_Rm_3_mfp_over_cell_20_mfp_limitX100/'

colors = []
colors += ['b']
colors += ['g']
colors += ['r']
colors += ['m']
colors += ['c']
colors += ['k']
colors += ['y']

plasma_modes = []
plasma_modes += ['isoT']
plasma_modes += ['isoTmfp']
plasma_modes += ['coold1']
plasma_modes += ['coold2']
plasma_modes += ['coold3']
# plasma_modes += ['cool']
# plasma_modes += ['cool_mfpcutoff']

# number_of_cells = 10
# number_of_cells = 15
# number_of_cells = 30
# number_of_cells = 40
# number_of_cells = 50
# number_of_cells = 70
number_of_cells = 100

# U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
U_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# U_list = [0, 0.2, 0.5, 0.75, 1.0]

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
            try:
                state, settings = load_simulation(state_file, settings_file)
                if state['successful_termination'] == True:
                    flux_list[ind_U] = state['flux_mean']
            except:
                pass

            # extract the density profile
            # chosen_plasma_mode = 'isoT'
            chosen_plasma_mode = 'isoTmfp'
            # chosen_plasma_mode = 'coold1'
            # chosen_plasma_mode = 'coold2'
            if plasma_mode == chosen_plasma_mode:
                plt.figure(2)
                # label = run_name
                label = 'U/$v_{th}$=' + str(U) + ', ' + define_LC_mode_label(LC_mode)
                plt.plot(state['n'], '-', label=label, linestyle=linestyle, color=colors[ind_U])

        # plot flux as a function of U
        # label_flux = plasma_modes[ind_mode] + '_N_' + str(number_of_cells) + '_' + LC_mode
        # label_flux = plasma_modes[ind_mode] + '_' + LC_mode
        label_flux = define_label(plasma_mode, LC_mode)
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

plt.figure(2)
plt.xlabel('cell number')
plt.ylabel('density [$m^{-3}$]')
plt.title('density profile (N=' + str(number_of_cells) + ' cells, plasma mode: ' + chosen_plasma_mode + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.grid(True)
