import matplotlib.pyplot as plt
import matplotlib.cm as cm

# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 14})

import numpy as np
from scipy.optimize import curve_fit

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation


def define_plasma_mode_label(plasma_mode):
    label = ''
    if plasma_mode == 'isoT':
        label += 'isothermal'
    elif plasma_mode == 'isoTmfp':
        # label += 'isothermal iso-mfp'
        label += 'diffusion'
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
# main_dir = '../runs/slurm_runs/set10_Rm_3_mfp_over_cell_0.04_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set11_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2/'
# main_dir = '../runs/slurm_runs/set12_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2_rbc_adjut_ntL_timestepdef_without_ntL/'
# main_dir = '../runs/slurm_runs/set13_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2_rbc_adjut_ntR/'

plasma_modes = []
plasma_modes += ['isoT']
plasma_modes += ['isoTmfp']
plasma_modes += ['coold1']
plasma_modes += ['coold2']
plasma_modes += ['coold3']
# plasma_modes += ['cool']
# plasma_modes += ['cool_mfpcutoff']

# num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# num_cells_list = [3, 5, 8]
num_cells_list = [3, 5, 8, 10, 15, 30, 50, 70, 100]

# colors = []
# colors += ['b']
# colors += ['g']
# colors += ['r']
# colors += ['m']
# colors += ['c']
colors = cm.rainbow(np.linspace(0, 1, len(num_cells_list)))

U = 0
# U = 0.05
# U = 0.1
# U = 0.2
# U = 0.3
# U = 0.5
# U = 0.75
# U = 1.0

linestyles = []
linestyles += ['-']
linestyles += ['--']

LC_modes = []
LC_modes += ['sLC']
# LC_modes += ['dLC']


for ind_mode in range(len(plasma_modes)):
    plasma_mode = plasma_modes[ind_mode]

    for ind_LC in range(len(LC_modes)):
        linestyle = linestyles[ind_LC]
        LC_mode = LC_modes[ind_LC]

        flux_list = np.nan * np.zeros(len(num_cells_list))
        for ind_N, number_of_cells in enumerate(num_cells_list):
            color = colors[ind_N]

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
                    flux_list[ind_N] = state['flux_mean']

                # label = run_name
                # label = 'N=' + str(number_of_cells) + ', ' + define_LC_mode_label(LC_mode)
                # label = define_label(plasma_mode, LC_mode)
                label = 'N=' + str(number_of_cells)
                plt.figure(1 + 2 * ind_mode)
                x = np.linspace(0, number_of_cells, number_of_cells)
                plt.plot(x, state['n'], '-', label=label, linestyle=linestyle, color=color)

                plt.figure(2 + 2 * ind_mode)
                x = np.linspace(0, 1, number_of_cells)
                plt.plot(x, state['n'], '-', label=label, linestyle=linestyle, color=color)

            except:
                pass

for ind_mode, mode in enumerate(plasma_modes):
    plt.figure(1 + 2 * ind_mode)
    plt.xlabel('cell number')
    plt.ylabel('density [$m^{-3}$]')
    # plt.title('mode "' + mode + '" density profiles $U/v_{th}$=' + str(U))
    # plt.title('mode "' + mode + '" density profiles')
    plt.title(define_plasma_mode_label(mode) + ' density profiles')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.grid(True)

    plt.figure(2 + 2 * ind_mode)
    plt.xlabel('cell number / N')
    plt.ylabel('density [$m^{-3}$]')
    plt.title(define_plasma_mode_label(mode) + ' density profiles')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.grid(True)
