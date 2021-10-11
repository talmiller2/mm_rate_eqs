import matplotlib

# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 14})

import numpy as np
from scipy.optimize import curve_fit

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from mm_rate_eqs.fusion_functions import get_lawson_parameters

from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss, get_magnetic_pressure, \
    get_ideal_gas_pressure, get_ideal_gas_energy_per_volume, get_magnetic_field_for_given_pressure, \
    get_bohm_diffusion_constant, get_larmor_radius


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

# main_dir += '/runs/slurm_runs/set2_Rm_3/'
# main_dir += '/runs/slurm_runs/set4_Rm_3_mfp_over_cell_4/'
# main_dir += '/runs/slurm_runs/set5_Rm_3_mfp_over_cell_20/'
# main_dir += '/runs/slurm_runs/set6_Rm_3_mfp_over_cell_1_mfp_limitX100/'
# main_dir += '/runs/slurm_runs/set7_Rm_3_mfp_over_cell_20_mfp_limitX100/'
# main_dir += '/runs/slurm_runs/set10_Rm_3_mfp_over_cell_0.04_mfp_limitX100/'
# main_dir += '/runs/slurm_runs/set14_MM_Rm_3_ni_2e22/'
# main_dir += '/runs/slurm_runs/set15_MM_Rm_3_ni_2e22_nend_1e-2_rbc_adjust_ntR/'
# main_dir += '/runs/slurm_runs/set16_MM_Rm_3_ni_4e23/'
# main_dir += '/runs/slurm_runs/set17_MM_Rm_3_ni_1e21/'
# main_dir += '/runs/slurm_runs/set20_MM_Rm_3_ni_2e22_trans_type_none/'
# main_dir += '/runs/slurm_runs/set21_MM_Rm_3_ni_2e22_trans_type_none_trans_fac_1/'
# main_dir += '/runs/slurm_runs/set22_MM_Rm_3_ni_1e21_trans_type_none/'
# main_dir += '/runs/slurm_runs/set24_MM_Rm_3_ni_2e20_trans_type_none/'
# main_dir += '/runs/slurm_runs/set25_MM_Rm_3_ni_4e23_trans_type_none/'
# main_dir += '/runs/slurm_runs/set26_MM_Rm_3_ni_2e20_trans_type_none_flux_cutoff_0.01/'
# main_dir += '/runs/slurm_runs/set27_MM_Rm_3_ni_2e22_trans_type_none_flux_cutoff_1e-3/'
# main_dir += '/runs/slurm_runs/set28_MM_Rm_3_ni_2e22_trans_type_none_flux_cutoff_1e-4/'
# main_dir += '/runs/slurm_runs/set29_MM_Rm_3_ni_2e20_trans_type_none_flux_cutoff_1e-4/'
# main_dir += '/runs/slurm_runs/set30_MM_Rm_3_ni_4e23_trans_type_none_flux_cutoff_1e-4/'
# main_dir += '/runs/slurm_runs/set31_MM_Rm_3_ni_1e21_trans_type_none_right_scat_fac_10/'
# main_dir += '/runs/slurm_runs/set32_MM_Rm_3_ni_1e21_trans_type_none_right_scat_fac_100/'
# main_dir += '/runs/slurm_runs/set33_MM_Rm_3_ni_1e21_trans_type_none_right_scat_fac_1/'
main_dir += '/runs/slurm_runs/set34_MM_Rm_10_ni_2e22_trans_type_none/'
# main_dir += '/runs/slurm_runs/set35_MM_Rm_10_ni_1e21_trans_type_none/'
# main_dir += '/runs/slurm_runs/set37_MM_Rm_10_ni_1e21_trans_type_none_flux_cutoff_1e-4/'


colors = []
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

num_cells_list = [3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# num_cells_list = [3, 5, 8]
# num_cells_list = [3, 5, 8, 10, 15, 30, 50, 70, 100]
# num_cells_list = [3, 5, 8, 10, 15, 20, 30, 50, 70, 100, 130, 150]

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

linewidth = 3

LC_modes = []
LC_modes += ['sLC']
# LC_modes += ['dLC']


for ind_mode in range(len(plasma_modes)):
    color = colors[ind_mode]
    plasma_mode = plasma_modes[ind_mode]

    for ind_LC in range(len(LC_modes)):
        linestyle = linestyles[ind_LC]
        # linestyle = '--'
        LC_mode = LC_modes[ind_LC]

        flux_list = np.nan * np.zeros(len(num_cells_list))
        n1_list = np.nan * np.zeros(len(num_cells_list))
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

                # post process the flux normalization
                # norm_factor = 2.0 * settings['cross_section_main_cell'] * settings['transmission_factor']
                # norm_factor = 2.0 * settings['cross_section_main_cell']
                # norm_factor *= state['n'][0] * state['v_th'][0]
                # norm_factor = state['n'][0] * state['v_th'][0]
                # state['flux_mean'] /= norm_factor
                ni = state['n'][0]
                Ti_keV = state['Ti'][0] / 1e3
                _, flux_lawson = get_lawson_parameters(ni, Ti_keV, settings)
                state['flux_mean'] *= settings['cross_section_main_cell']
                state['flux_mean'] /= flux_lawson

                # if state['successful_termination'] == True:
                #     flux_list[ind_N] = state['flux_mean']
                #     n1_list[ind_N] = state['n'][-1]

                flux_list[ind_N] = state['flux_mean']
                n1_list[ind_N] = state['n'][-1]
                # n1_list[ind_N] = state['n'][-2]

            except:
                pass

            # # extract the density profile
            # chosen_num_cells = 30
            # if number_of_cells == chosen_num_cells:
            #     plt.figure(2)
            #     # label = run_name
            #     # label = 'N=' + str(number_of_cells) + ', ' + define_LC_mode_label(LC_mode)
            #     # label = define_label(plasma_mode, LC_mode)
            #     label = define_plasma_mode_label(plasma_mode)
            #     plt.plot(state['n'], '-', label=label, linestyle=linestyle, color=color)

        # plot flux as a function of N
        # flux_list /= flux_norm
        # flux_list /= flux_list[3]
        # label_flux = plasma_modes[ind_mode] + '_U_' + str(U) + '_' + LC_mode
        # label_flux = plasma_modes[ind_mode] + ', mfp/l=4'
        # label_flux = define_label(plasma_mode, LC_mode)
        label_flux = define_plasma_mode_label(plasma_mode)
        plt.figure(1)
        plt.plot(num_cells_list, flux_list, label=label_flux, linestyle=linestyle, color=color,
                 # plt.plot(num_cells_list, n1_list, '-', label=label_flux, linestyle=linestyle, color=color,
                 linewidth=linewidth)

        # remove some of the values prior to fit
        # ind_min = 0
        # ind_min = 5
        # ind_max = len(num_cells_list)
        # # if plasma_mode == 'coold1':
        # #     ind_max -= 4
        # num_cells_list_for_fit = num_cells_list[ind_min:ind_max]
        # flux_list_for_fit = flux_list[ind_min:ind_max]

        # # clear nans for fit
        # norm_factor = np.nanmax(flux_list_for_fit)
        # # norm_factor = 1e27
        # num_cells_list_for_fit = np.array(num_cells_list_for_fit)
        # inds_flux_not_nan = [i for i in range(len(flux_list_for_fit)) if not np.isnan(flux_list_for_fit[i])]
        # n_cells = num_cells_list_for_fit[inds_flux_not_nan]
        # flux_cells = flux_list_for_fit[inds_flux_not_nan] / norm_factor
        # # fit_function = lambda x, a, b, gamma: a + b / x ** gamma
        # # fit_function = lambda x, b, gamma: b / x ** gamma
        # fit_function = lambda x, b, gamma: b * x ** gamma
        # # fit_function = lambda x, b: b / x
        # popt, pcov = curve_fit(fit_function, n_cells, flux_cells)
        # # flux_cells_fit = fit_function(n_cells, *popt) * norm_factor
        # flux_cells_fit = fit_function(num_cells_list, *popt) * norm_factor
        # # print('popt:' + str(popt))
        # # label = 'fit decay power: ' + '{:0.3f}'.format(popt[-1])
        # label = 'fit power: ' + '{:0.3f}'.format(popt[-1])
        # # plt.plot(n_cells, flux_cells_fit, label=label, linestyle='--', color=color)
        # # plt.plot(num_cells_list, flux_cells_fit, label=label, linestyle='--', color=color, linewidth=linewidth)

        # plot prediction of analytic the diffusion model
        n0 = state['n'][0]
        # n1 = n0 * 0.2
        # n1 = n0 * 0.5
        # n1 = n0 * 0.01
        n1 = np.array(n1_list)
        D0 = state['mean_free_path'][0] * state['v_th'][0]
        L = np.array(num_cells_list) * state['mirror_cell_sizes'][0]
        lambda_over_l = '{:.2f}'.format(state['mean_free_path'][0] / state['mirror_cell_sizes'][0])
        pre_factor = n0 * D0 / L
        # pre_factor /= state['n'][0] * state['v_th'][0]
        if plasma_mode == 'isoTmfp':
            # linear diffusion
            flux_analytic = pre_factor * (1 - n1 / n0)
        elif plasma_mode == 'isoT':
            flux_analytic = pre_factor * np.log(n0 / n1)
        elif 'cool' in plasma_mode:
            d = int(plasma_mode[-1]) * 1.0
            flux_analytic = - pre_factor * d / 5 * ((n1 / n0) ** (5 / d) - 1)
        # ind_flux_fit = 6
        num_cells_for_fit = 20
        ind_flux_fit = np.where(np.array(num_cells_list) >= num_cells_for_fit)[0][0]
        flux_norm = flux_list[ind_flux_fit]
        flux_analytic /= flux_analytic[ind_flux_fit]
        flux_analytic *= flux_norm
        # plt.plot(num_cells_list, n1_list,
        plt.plot(num_cells_list, flux_analytic,
                 # label='analytic theory',
                 label=None,
                 linestyle='dashdot',
                 # linestyle='-.',
                 color=color,
                 linewidth=2)
        # plt.title('rate eqs vs theory ($\\lambda/l=$' + lambda_over_l + ', $n_1/n_0=$' + str(n1 / n0) + ')')
        # plt.title('rate eqs vs theory ($\\lambda/l=$' + lambda_over_l + ', $n_1$ from rate eqs)')

        # plt.figure(2)
        # plt.plot(num_cells_list, n1_list, '-', label=label_flux, linestyle=linestyle, color=color, linewidth=linewidth)

# plot a 1/N reference line
# const = 1.1
# const = 14
# plt.plot(num_cells_list, const / np.array(num_cells_list), '-', label='$1/N$ reference', linestyle='--', color='k',
#          linewidth=linewidth)


# add plot for then radial flux in the MM section alone

B = 10.0  # T
D_bohm = get_bohm_diffusion_constant(state['Te'][0], B)  # [m^2/s]
# integral of dn/dz for linearly declining n is n*L/2
dndx = state['n'][0] * np.ones(len(num_cells_list)) / 2 / (settings['diameter_main_cell'] / 2)
radial_flux_density = D_bohm * dndx
system_total_length = np.array(num_cells_list) * settings['cell_size']
cyllinder_radial_cross_section = np.pi * settings['diameter_main_cell'] * system_total_length
radial_flux_bohm = radial_flux_density * cyllinder_radial_cross_section
radial_flux_bohm /= flux_lawson

gyro_radius = get_larmor_radius(state['Ti'][0], B)
D_classical = gyro_radius ** 2 * state['coulomb_scattering_rate'][0]
dndx = state['n'][0] * np.ones(len(num_cells_list)) / 3 / (settings['diameter_main_cell'] / 2)
radial_flux_density = D_classical * dndx
radial_flux_classical = radial_flux_density * cyllinder_radial_cross_section
radial_flux_classical /= flux_lawson

# plt.plot(num_cells_list, radial_flux_bohm, label='radial bohm', linestyle='-.', color='k', linewidth=linewidth)
# plt.plot(num_cells_list, radial_flux_classical, label='radial classical', linestyle=':', color='k', linewidth=linewidth)

fig = plt.figure(1)
plt.yscale("log")
plt.xscale("log")
plt.xlabel('N')
# plt.ylabel('flux [$s^{-1}$]')
# plt.ylabel('$\\phi_{p}$ [$m^{-2}s^{-1}$]')
# plt.ylabel('$\\phi_{p} / \\phi_{p,0}$')
plt.ylabel('$\\phi_{ss} / \\phi_{lawson}$')
# plt.title('flux as a function of system size')
# plt.title('flux as a function of system size ($U/v_{th}$=' + str(U) + ')')
plt.tight_layout()
plt.grid(True)
plt.legend()

text = '(a)'
# text = '(b)'
plt.text(0.98, 0.97, text, fontdict={'fontname': 'times new roman', 'weight': 'bold', 'size': 20},
         horizontalalignment='right', verticalalignment='top',
         transform=fig.axes[0].transAxes)

ax = plt.gca()
ax.set_xticks([10, 100])
ax.set_yticks([10, 100])
# ax.set_yticks([1000, 2000, 4000])
from matplotlib.ticker import StrMethodFormatter, NullFormatter

ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_minor_formatter(NullFormatter())

# plt.figure(2)
# # plt.xlabel('cell number')
# plt.xlabel('N')
# # plt.ylabel('density [$m^{-3}$]')
# plt.ylabel('$n_1$ [$m^{-3}$]')
# # plt.title('density profile (N=' + str(chosen_num_cells) + ')')
# # plt.title('density profile (N=' + str(chosen_num_cells) + ' cells, $U/v_{th}$=' + str(U) + ')')
# plt.tight_layout()
# plt.grid(True)
# plt.legend()

# test
# flux_list1 = flux_list
# flux_list2 = flux_list
# plt.figure(3)
# plt.plot(num_cells_list, flux_list1 / flux_list2)
# plt.xlabel('N')
# plt.tight_layout()
# plt.grid(True)
# plt.legend()

# save pics in high res
# save_dir = '../../../Papers/texts/paper2020/pics/'
# save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics/'
save_dir = '/Users/talmiller/Dropbox/UNI/Courses Graduate/Plasma/Papers/texts/paper2020/pics_with_Rm_10/'

# file_name = 'flux_function_of_N'
# # file_name = 'flux_function_of_N_suboptimal'
# beingsaved = plt.figure(1)
# beingsaved.savefig(save_dir + file_name + '.eps', format='eps')
