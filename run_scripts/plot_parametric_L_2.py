import matplotlib.pyplot as plt

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
# main_dir = '../runs/slurm_runs/set6_Rm_3_mfp_over_cell_1_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set7_Rm_3_mfp_over_cell_20_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set10_Rm_3_mfp_over_cell_0.04_mfp_limitX100/'
main_dir = '../runs/slurm_runs/set14_MM_Rm_3_ni_2e22/'
# main_dir = '../runs/slurm_runs/set15_MM_Rm_3_ni_2e22_nend_1e-2_rbc_adjust_ntR/'
# main_dir = '../runs/slurm_runs/set16_MM_Rm_3_ni_4e23/'
# main_dir = '../runs/slurm_runs/set17_MM_Rm_3_ni_1e21/'

colors = []
colors += ['b']
colors += ['g']
colors += ['r']
colors += ['m']
colors += ['c']

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
plasma_modes += ['coold1']
plasma_modes += ['coold2']
plasma_modes += ['coold3']
# plasma_modes += ['cool']
# plasma_modes += ['cool_mfpcutoff']

# num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# num_cells_list = [3, 5, 8]
# num_cells_list = [3, 5, 8, 10, 15, 30, 50, 70, 100]
num_cells_list = [3, 5, 8, 10, 15, 20, 30, 50, 70, 100, 130, 150]

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
            try:
                state, settings = load_simulation(state_file, settings_file)
                if state['successful_termination'] == True:
                    flux_list[ind_N] = state['flux_mean']
            except:
                pass


            # extract the density profile
            chosen_num_cells = 30
            if number_of_cells == chosen_num_cells:
                plt.figure(2)
                # label = run_name
                # label = 'N=' + str(number_of_cells) + ', ' + define_LC_mode_label(LC_mode)
                # label = define_label(plasma_mode, LC_mode)
                label = define_plasma_mode_label(plasma_mode)
                plt.plot(state['n'], '-', label=label, linestyle=linestyle, color=color)

        # plot flux as a function of N
        # label_flux = plasma_modes[ind_mode] + '_U_' + str(U) + '_' + LC_mode
        # label_flux = plasma_modes[ind_mode] + ', mfp/l=4'
        # label_flux = define_label(plasma_mode, LC_mode)
        label_flux = define_plasma_mode_label(plasma_mode)
        plt.figure(1)
        plt.plot(num_cells_list, flux_list, '-', label=label_flux, linestyle=linestyle, color=color)
        plt.yscale("log")

        # # clear nans for fit
        norm_factor = 1e27
        num_cells_list = np.array(num_cells_list)
        inds_flux_not_nan = [i for i in range(len(flux_list)) if not np.isnan(flux_list[i])]
        n_cells = num_cells_list[inds_flux_not_nan]
        flux_cells = flux_list[inds_flux_not_nan] / norm_factor
        # fit_function = lambda x, a, b, gamma: a + b / x ** gamma
        fit_function = lambda x, b, gamma: b / x ** gamma
        # fit_function = lambda x, b: b / x
        popt, pcov = curve_fit(fit_function, n_cells, flux_cells)
        flux_cells_fit = fit_function(n_cells, *popt) * norm_factor
        # plt.plot(n_cells, flux_cells_fit, label=label + ' fit', linestyle=':', color=color)
        plt.plot(n_cells, flux_cells_fit, label='fit power = ' + '{:0.3f}'.format(popt[-1]), linestyle='--',
                 color=color)

plt.figure(1)
plt.xlabel('N')
plt.ylabel('flux [$s^{-1}$]')
# plt.title('flux as a function of system size')
plt.title('flux as a function of system size ($U/v_{th}$=' + str(U) + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(2)
plt.xlabel('cell number')
plt.ylabel('density [$m^{-3}$]')
# plt.title('density profile (N=' + str(chosen_num_cells) + ')')
plt.title('density profile (N=' + str(chosen_num_cells) + ' cells, $U/v_{th}$=' + str(U) + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.grid(True)
