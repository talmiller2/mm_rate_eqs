import matplotlib

# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

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


# plt.close('all')

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs/'

# main_dir = '../runs/slurm_runs/set2_Rm_3/'
# main_dir = '../runs/slurm_runs/set4_Rm_3_mfp_over_cell_4/'
# main_dir = '../runs/slurm_runs/set5_Rm_3_mfp_over_cell_20/'
# main_dir = '../runs/slurm_runs/set6_Rm_3_mfp_over_cell_1_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set7_Rm_3_mfp_over_cell_20_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set10_Rm_3_mfp_over_cell_0.04_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set11_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2/'
# main_dir = '../runs/slurm_runs/set12_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2_rbc_adjut_ntL_timestepdef_without_ntL/'
# main_dir = '../runs/slurm_runs/set13_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2_rbc_adjut_ntR/'
# main_dir = '../runs/slurm_runs/set14_MM_Rm_3_ni_2e22/'
# main_dir = '../runs/slurm_runs/set15_MM_Rm_3_ni_2e22_nend_1e-2_rbc_adjust_ntR/'
# main_dir = '../runs/slurm_runs/set16_MM_Rm_3_ni_4e23/'
# main_dir = '../runs/slurm_runs/set17_MM_Rm_3_ni_1e21/'
# main_dir = '../runs/slurm_runs/set20_MM_Rm_3_ni_2e22_trans_type_none/'
# main_dir = '../runs/slurm_runs/set21_MM_Rm_3_ni_2e22_trans_type_none_trans_fac_1/'
# main_dir += '/runs/slurm_runs/set22_MM_Rm_3_ni_1e21_trans_type_none/'
# main_dir = '../runs/slurm_runs/set24_MM_Rm_3_ni_2e20_trans_type_none/'
# main_dir = '../runs/slurm_runs/set25_MM_Rm_3_ni_4e23_trans_type_none/'
# main_dir = '../runs/slurm_runs/set26_MM_Rm_3_ni_2e20_trans_type_none_flux_cutoff_0.01/'
# main_dir = '../runs/slurm_runs/set27_MM_Rm_3_ni_2e22_trans_type_none_flux_cutoff_1e-3/'
# main_dir += '/runs/slurm_runs/set28_MM_Rm_3_ni_2e22_trans_type_none_flux_cutoff_1e-4/'
# main_dir = '../runs/slurm_runs/set29_MM_Rm_3_ni_2e20_trans_type_none_flux_cutoff_1e-4/'
# main_dir = '../runs/slurm_runs/set30_MM_Rm_3_ni_4e23_trans_type_none_flux_cutoff_1e-4/'
main_dir += '/runs/slurm_runs/set34_MM_Rm_10_ni_2e22_trans_type_none/'
# main_dir += '/runs/slurm_runs/set35_MM_Rm_10_ni_1e21_trans_type_none/'
# main_dir += '/runs/slurm_runs/set37_MM_Rm_10_ni_1e21_trans_type_none_flux_cutoff_1e-4/'

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
plasma_modes += ['coold1']
plasma_modes += ['coold2']
plasma_modes += ['coold3']
# plasma_modes += ['cool']
# plasma_modes += ['cool_mfpcutoff']

# colors = cm.rainbow(np.linspace(0, 1, len(plasma_modes)))
# colors = ['b', 'g', 'r', 'm', 'c', 'k']
# colors = ['b', 'g', 'r', 'k', 'm', 'c']
# colors = ['b', 'g', 'r', 'k', 'orange', 'c']
colors = ['k', 'b', 'g', 'orange', 'r']

linewidth = 3

# number_of_cells = 5
# number_of_cells = 10
number_of_cells = 30
# number_of_cells = 70
# number_of_cells = 100
# number_of_cells = 150
U = 0

for ind_mode in range(len(plasma_modes)):
    plasma_mode = plasma_modes[ind_mode]
    color = colors[ind_mode]

    run_name = plasma_mode
    run_name += '_N_' + str(number_of_cells) + '_U_' + str(U)
    run_name += '_sLC'
    # print('run_name = ' + run_name)

    save_dir = main_dir + '/' + run_name

    state_file = save_dir + '/state.pickle'
    settings_file = save_dir + '/settings.pickle'
    try:
        state, settings = load_simulation(state_file, settings_file)
        print('plasma_mode:', plasma_mode, ', successful_termination:', state['successful_termination'])
        label = define_plasma_mode_label(plasma_mode)

        print('flux_norm = ' + str(state['flux_mean'] / (state['n'][0] * state['v_th'][0])))

        # plt.figure(1)
        # x = np.linspace(0, number_of_cells, number_of_cells)
        # n0 = settings['n0']
        # plt.plot(x, state['n_c'] / n0, label='$n_{c}$ ' + label, linestyle='solid', color=color)
        # plt.plot(x, state['n_tR'] / n0, label='$n_{tR}$ ' + label, linestyle='dashed', color=color)
        # plt.plot(x, state['n_tL'] / n0, label='$n_{tL}$ ' + label, linestyle='dashdot', color=color)
        # plt.xlabel('cell number')
        # # plt.ylabel('[$m^{-3}$]')
        # plt.ylabel('$n/n_{i,0}$')
        # plt.title('density profiles for N=' + str(number_of_cells))
        # plt.tight_layout()
        # plt.grid(True)
        # plt.legend()

        x = np.linspace(0, number_of_cells, number_of_cells)
        n0 = settings['n0']
        print('n0 = ', n0)

        # plt.figure(1)
        # plt.subplot(2, 1, 1)
        fig = plt.figure(10)
        # plt.plot(x, state['n_c'] / n0, label='$n_{c}$ ' + label, linestyle='solid', color=color, linewidth=linewidth)
        plt.plot(x, state['n_c'] / n0, label=label, linestyle='solid', color=color, linewidth=linewidth)
        plt.xlabel('cell number')
        # plt.ylabel('[$m^{-3}$]')
        plt.ylabel('$n/n_{i,0}$')
        # plt.title('density profiles for N=' + str(number_of_cells))
        plt.tight_layout()
        plt.grid(True)
        plt.legend()


        # plt.subplot(2, 1, 2)
        plt.figure(11)
        # plt.plot(x, state['n_tR'] / n0, label='$n_{tR}$', linestyle='solid', color=color, linewidth=linewidth)
        # plt.plot(x, state['n_tL'] / n0, label='$n_{tL}$', linestyle='dashed', color=color, linewidth=linewidth)
        plt.plot(x, state['n_tR'] / n0, linestyle='solid', color=color, linewidth=linewidth)
        plt.plot(x, state['n_tL'] / n0, linestyle='dashed', color=color, linewidth=linewidth)
        plt.xlabel('cell number')
        # plt.ylabel('[$m^{-3}$]')
        plt.ylabel('$n/n_{i,0}$')
        # plt.title('density profiles for N=' + str(number_of_cells))
        # plt.tight_layout()
        plt.grid(True)
        # plt.legend()
        plt.tight_layout()

        fig = plt.figure(2)
        x = np.linspace(0, number_of_cells, number_of_cells)
        plt.plot(x, state['n'] / n0, label=label, linestyle='solid', color=color, linewidth=linewidth)
        plt.xlabel('cell number')
        plt.ylabel('$n/n_{i,0}$')
        # plt.title('density profile for N=' + str(number_of_cells))
        plt.tight_layout()
        plt.grid(True)
        plt.legend()
        # plt.yscale('log')

        # ## for analytic form to numeric density solution
        # # if 'cool' in plasma_mode:
        # #     d = int(plasma_mode[-1])
        # #     fit_function = lambda x, a: (1 + a * x) ** (d / 5.0)
        # #     if d == 1: a_guess = -0.02
        # #     if d == 2: a_guess = -0.02
        # #     if d == 3: a_guess = -0.02
        # # elif plasma_mode == 'isoT':
        # #     fit_function = lambda x, a: np.exp(-a * x)
        # #     a_guess = 0.025
        # # else:
        # #     fit_function = lambda x, a: 1 - a * x
        # #     a_guess = 0.01
        # #
        # # n_normed = state['n'] / n0
        # # popt, pcov = curve_fit(fit_function, x, n_normed)
        # # print('popt', popt)
        # # print('pcov', pcov)
        # # n_normed_fit = fit_function(x, *popt)
        # # plt.plot(x, n_normed_fit, label='fit', linestyle='dashdot', color=color, linewidth=linewidth)
        # # n_normed_guess_fit = fit_function(x, a_guess)
        # # # plt.plot(x, n_normed_guess_fit, label='guess', linestyle='dashdot', color=color, linewidth=linewidth)
        # # plt.legend()
        #
        # plot analytic model
        n0 = 1.0
        # n1 = n0 * 0.2
        n1 = state['n'][-1] / state['n'][0]
        L = x * state['mirror_cell_sizes'][0]
        lambda_over_l = '{:.2f}'.format(state['mean_free_path'][0] / state['mirror_cell_sizes'][0])
        n1_over_n0 = '{:.2f}'.format(n1 / n0)
        if plasma_mode == 'isoTmfp':
            # linear diffusion
            n_analytic = n0 * (1 - x / number_of_cells) + n1 * x / number_of_cells
        elif plasma_mode == 'isoT':
            n_analytic = n0 * (n0 / n1) ** (- x / number_of_cells)
        elif 'cool' in plasma_mode:
            d = int(plasma_mode[-1]) * 1.0
            n_analytic = n0 * (1 + ((n1 / n0) ** (5 / d) - 1) * x / number_of_cells) ** (d / 5)
        plt.figure(2)
        plt.plot(x, n_analytic, linestyle='dashdot', color=color, linewidth=2)
        # plt.title('rate eqs vs theory ($\\lambda/l=$' + lambda_over_l + ', $n_1/n_0=$' + n1_over_n0 + ')')
        # plt.title('rate eqs vs theory ($\\lambda/l=$' + lambda_over_l + ')')

        plt.figure(3)
        x = np.linspace(0, number_of_cells, number_of_cells)
        plt.plot(x, state['mean_free_path'] / settings['cell_size'], label=label, linestyle='solid', color=color,
                 linewidth=linewidth)
        plt.yscale('log')
        plt.xlabel('cell number')
        plt.ylabel('$\\lambda/l$')
        # plt.title('$\\lambda/l$ profiles for N=' + str(number_of_cells))
        plt.tight_layout()
        plt.grid(True)
        plt.legend()

        if ind_mode == len(plasma_modes) - 1:
            # plt.text(0.05, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
            plt.text(0.08, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
                     horizontalalignment='left', verticalalignment='top',
                     transform=fig.axes[0].transAxes)

        # plt.figure(4)
        # x = np.linspace(0, number_of_cells, number_of_cells)
        # plt.plot(x, state['flux'], label=label, linestyle='solid', color=color, linewidth=linewidth)
        # # phi0 = n0 * state['v_th'][0]
        # # plt.plot(x, state['flux'] / phi0, label=label, linestyle='solid', color=color, linewidth=linewidth)
        # # plt.yscale('log')
        # plt.xlabel('cell number')
        # plt.ylabel('$\\phi/\\phi_0$')
        # # plt.title('flux profile for N=' + str(number_of_cells))
        # plt.tight_layout()
        # plt.grid(True)
        # plt.legend()

        # plt.figure(5)
        # plt.plot((state['n_tR'] - state['n_tL']) / n0, label='$n_{tR}-n_{tL}$ ' + label, linestyle='solid', color=color,
        #          linewidth=linewidth)
        # plt.plot((state['n_tR'][:-1] - state['n_tL'][1:]) / n0, label='$n_{tR}-n_{tL}$ neighbours ' + label,
        #          linestyle='dashed', color=color, linewidth=linewidth)
        # # diff = np.zeros(len(x)-1)
        # # for i in range(len(diff)):
        # #     diff[i] = (state['n_tR'][i] - state['n_tL'][i+1]) / n0
        # # plt.plot(diff, label='$n_{tR}-n_{tL}$ neighbours ' + label, linestyle='dashdot', color=color)
        # plt.xlabel('cell number')
        # plt.ylabel('$n/n_{i,0}$')
        # # plt.title('$n_{tR}-n_{tL}$ for N=' + str(number_of_cells))
        # plt.tight_layout()
        # plt.grid(True)
        # plt.legend()
        #
        # plt.figure(6)
        # # plt.plot(x, state['v_th'], label=label, linestyle='solid', color=color)
        # plt.plot(x, state['v_th'] / state['v_th'][0], label=label, linestyle='solid', color=color, linewidth=linewidth)
        # plt.xlabel('cell number')
        # # plt.ylabel('$v_{th}$')
        # plt.ylabel('$v_{th}/v_{th,0}$')
        # # plt.title('thermal velocity profiles for N=' + str(number_of_cells))
        # plt.tight_layout()
        # plt.grid(True)
        # plt.legend()

    except:
        pass

##### add text for paper
fig = plt.figure(10)
plt.text(0.98, 0.97, '(a)', fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='right', verticalalignment='top',
         transform=fig.axes[0].transAxes)
fig = plt.figure(11)
plt.text(0.98, 0.97, '(b)', fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='right', verticalalignment='top',
         transform=fig.axes[0].transAxes)

text = '(a)'
# text = '(b)'

fig = plt.figure(2)
plt.text(0.98, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='right', verticalalignment='top',
         transform=fig.axes[0].transAxes)

fig = plt.figure(3)
plt.text(0.04, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='left', verticalalignment='top',
         transform=fig.axes[0].transAxes)

ax = plt.gca()
# ax.set_yticks([1, 10])
# ax.set_yticks([5, 10, 20, 30])
ax.set_yticks([0.1, 1, 10])
from matplotlib.ticker import StrMethodFormatter, NullFormatter

ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
ax.yaxis.set_minor_formatter(NullFormatter())

##### save pics in high res
# save_dir = '../../../Papers/texts/paper2020/pics/'
# save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics/'
save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics_with_Rm_10/'

# file_name = 'density_profiles_N_30_nc'
# # file_name = 'density_profiles_N_100_nc'
# # file_name = 'density_profiles_N_30_nc_suboptimal'
# # file_name = 'density_profiles_N_100_nc_suboptimal'
# beingsaved = plt.figure(10)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
# #
# file_name = 'density_profiles_N_30_nr_nl'
# # file_name = 'density_profiles_N_100_nr_nl'
# # file_name = 'density_profiles_N_30_nr_nl_suboptimal'
# # file_name = 'density_profiles_N_100_nr_nl_suboptimal'
# beingsaved = plt.figure(11)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')

# file_name = 'density_profiles_N_30_with_theory'
# file_name = 'density_profiles_N_30_suboptimal_with_theory'
# beingsaved = plt.figure(2)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')

file_name = 'mfp_profiles_N_30'
# file_name = 'mfp_profiles_N_30_suboptimal'
# file_name = 'mfp_profiles_N_100'
# beingsaved = plt.figure(3)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
