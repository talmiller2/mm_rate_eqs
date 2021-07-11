import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

import matplotlib

# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from scipy.optimize import curve_fit
from mm_rate_eqs.fusion_functions import get_lawson_parameters


def define_plasma_mode_label(plasma_mode):
    label = ''
    if plasma_mode == 'isoT':
        label += 'isothermal'
    elif plasma_mode == 'isoTmfp':
        # label += 'isothermal iso-mfp'
        # label += 'diffusion'
        # label += 'linear diffusion'
        label += 'constant diffusion'
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

# main_dir = '/Users/talmiller/Downloads/mm_rate_eqs/run_scripts/'
main_dir = '/Users/talmiller/Downloads/mm_rate_eqs/'

# main_dir = '../runs/slurm_runs/set3_N_20/'
# main_dir = '../runs/slurm_runs/set8_N_30_mfp_over_cell_1_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set9_N_30_mfp_over_cell_40_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set18_MM_N_30_ni_2e22/'
# main_dir = '../runs/slurm_runs/set19_MM_N_30_ni_1e21/'
# main_dir += '../runs/slurm_runs/set23_MM_N_30_ni_2e22_trans_type_none/'
main_dir += '/runs/slurm_runs/set23_MM_N_30_ni_2e22_trans_type_none/'

# colors = []
# colors += ['b']
# colors += ['g']
# colors += ['r']
# colors += ['m']
# colors += ['k']
# colors += ['c']
colors = ['k', 'b', 'g', 'orange', 'r']

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
plasma_modes += ['coold1']
plasma_modes += ['coold2']
plasma_modes += ['coold3']
# plasma_modes += ['cool']
# plasma_modes += ['cool_mfpcutoff']

number_of_cells = 30
# number_of_cells = 20
U = 0
# U = 0.1
# U = 0.3
# U = 0.5
# Rm_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# Rm_list = [1.1, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0]
Rm_list = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0]
# Rm_list = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0]
# Rm_list = [1.1, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
# Rm_list = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0]

linestyles = []
linestyles += ['-']
linestyles += ['--']

linewidth = 3

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
            try:
                state, settings = load_simulation(state_file, settings_file)

                # post process the flux normalization
                # norm_factor = 2.0 * settings['cross_section_main_cell'] * settings['transmission_factor']
                # norm_factor *= state['n'][0] * state['v_th'][0]
                # state['flux_mean'] /= norm_factor
                ni = state['n'][0]
                Ti_keV = state['Ti'][0] / 1e3
                _, flux_lawson = get_lawson_parameters(ni, Ti_keV, settings)
                state['flux_mean'] *= settings['cross_section_main_cell']
                state['flux_mean'] /= flux_lawson

                if state['successful_termination'] == True:
                    flux_list[ind_Rm] = state['flux_mean']
            except:
                pass

        # plot flux as a function of U
        # label_flux = plasma_modes[ind_mode] + '_N_' + str(number_of_cells) + '_' + LC_mode
        # label_flux = plasma_modes[ind_mode] + '_' + LC_mode
        # label_flux = plasma_modes[ind_mode] + '_U_' + str(U) + '_' + LC_mode
        # label_flux = define_label(plasma_mode, LC_mode)
        label_flux = define_plasma_mode_label(plasma_mode)
        plt.figure(1)
        plt.plot(Rm_list, flux_list, '-', label=label_flux, linestyle=linestyle, color=color, linewidth=linewidth)

        # remove some of the values prior to fit
        # # ind_min = 0
        # # ind_min = 3
        # ind_min = 5
        # Rm_list_for_fit = Rm_list[ind_min:]
        # flux_list_for_fit = flux_list[ind_min:]
        #
        # # clear nans for fit
        # # norm_factor = 1e27
        # norm_factor = np.nanmax(flux_list_for_fit)
        # # norm_factor = 1
        # Rm_list_for_fit = np.array(Rm_list_for_fit)
        # inds_flux_not_nan = [i for i in range(len(flux_list_for_fit)) if not np.isnan(flux_list_for_fit[i])]
        # Rm_cells = Rm_list_for_fit[inds_flux_not_nan]
        # flux_cells = flux_list_for_fit[inds_flux_not_nan] / norm_factor
        # # fit_function = lambda x, a, b, gamma: a + b / x ** gamma
        # # fit_function = lambda x, b, gamma: b / x ** gamma
        # fit_function = lambda x, b, gamma: b * x ** gamma
        # # fit_function = lambda x, b: b / x
        # popt, pcov = curve_fit(fit_function, Rm_cells, flux_cells)
        # # flux_cells_fit = fit_function(Rm_cells, *popt) * norm_factor
        # flux_cells_fit = fit_function(Rm_list, *popt) * norm_factor
        # # label = 'fit decay power: ' + '{:0.3f}'.format(popt[-1])
        # label = 'fit power: ' + '{:0.3f}'.format(popt[-1])
        # # plt.plot(Rm_cells, flux_cells_fit, label=label, linestyle='--', color=color)
        # # plt.plot(Rm_list, flux_cells_fit, label=label, linestyle='--', color=color, linewidth=linewidth)

# plot a 1/Rm reference line
const = 2e2
# const = 0.75e27
# const = 14
# const = 14
plt.plot(Rm_list, const / np.array(Rm_list), '-', label='$1/R_m$ reference', linestyle='--', color='k',
         linewidth=2)

plt.figure(1)
plt.yscale("log")
# plt.xscale("log")
plt.xlabel('$R_m$')
# plt.ylabel('flux [$s^{-1}$]')
# plt.ylabel('$\\phi_{p}$ [$m^{-2}s^{-1}$]')
# plt.ylabel('$\\phi_{p} / \\phi_{p,0}$')
plt.ylabel('$\\phi_{ss} / \\phi_{lawson}$')
# plt.title('flux as a function of mirror ratio (N=' + str(number_of_cells) + ', $U/v_{th}$=' + str(U) + ')')
# plt.title('flux as a function of mirror ratio (N=' + str(number_of_cells) + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()

# save pics in high res
# save_dir = '../../../Papers/texts/paper2020/pics/'
save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics/'

# file_name = 'flux_function_of_Rm'
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
