import matplotlib
# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib import cm

# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 18})
# plt.rcParams.update({'font.size': 12})

import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from mm_rate_eqs.fusion_functions import get_lawson_parameters, get_lawson_criterion_piel
from mm_rate_eqs.plot_functions import update_format_coord

plt.close('all')

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs//runs/slurm_runs/'
main_dir += 'set57_MMM_ni_1e21_Ti_10keV_constmfp'

# num_cells = 10
# num_cells = 30
num_cells = 50
# num_cells = 80

# linewidth = 1
linewidth = 2

# axes_label_size = 12
axes_label_size = 18
title_fontsize = 12

# cmap = 'viridis'
# cmap = 'plasma'
# cmap = 'inferno'
cmap = 'coolwarm'

# gas_name = 'deuterium'
gas_name = 'tritium'

# num_cells_list = [10, 30, 50, 80]
num_cells_list = [50]

# scat_factor_list = [0.1, 1]
# # scat_factor_list = [0.1]
scat_factor_list = [1]

scat_asym_list = [0.5, 1, 2]
# scat_asym_list = [0.5]
# scat_asym_list = [1]
# scat_asym_list = [2]


for num_cells in num_cells_list:
    for scat_factor in scat_factor_list:
        for scat_asym_factor in scat_asym_list:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))

            compiled_set_name = 'compiled_'
            compiled_set_name += 'N_' + str(num_cells)
            compiled_set_name += '_scatfac_' + str(scat_factor)
            compiled_set_name += '_scatasym_' + str(scat_asym_factor)
            compiled_set_name += '_' + gas_name
            print(compiled_set_name)

            # load data of flux for all alpha,beta
            compiled_save_file = main_dir + '/' + compiled_set_name + '.mat'

            mat_dict = loadmat(compiled_save_file)
            flux_mat = mat_dict['flux_mat']
            Rm_list = mat_dict['Rm_list'][0]
            U_list = mat_dict['U_list'][0]

            # load on of the settings files
            # main_dir_dir_settings = '/Users/talmiller/Downloads/mm_rate_eqs//runs/slurm_runs/'
            # main_dir_dir_settings += 'set50_MM_Rm_10_ni_1e20_Ti_10keV_withRMF_zeroRL_fluxeps1e-2'
            main_dir_dir_settings = main_dir
            state_file = main_dir_dir_settings + '/state.pickle'
            settings_file = main_dir_dir_settings + '/settings.pickle'
            state, settings = load_simulation(state_file, settings_file)

            # post process the flux normalization
            ni = state['n'][0]
            # ni = 1e21 # phi_lawson~n^2 while phi_rate~n, so higher n means it is easier to read lawson.
            Ti_keV = state['Ti'][0] / 1e3

            _, flux_lawson_ignition_origial = get_lawson_parameters(ni, Ti_keV, settings)
            _, flux_lawson_piel, _, flux_lawson_ignition_piel = get_lawson_criterion_piel(ni, Ti_keV, settings)
            cross_section_main_cell = settings['cross_section_main_cell']
            v_th = state['v_th'][0]
            flux_single_naive = ni * v_th
            # print('ni=', ni, 'Ti_keV=', Ti_keV, 'flux_single_naive=', '{:.2e}'.format(flux_single_naive), 'flux_lawson_ignition_piel=', '{:.2e}'.format(flux_lawson_ignition_piel))

            ## plot flux 2d map
            X, Y = np.meshgrid(Rm_list, U_list)
            x_label = '$R_m$'
            y_label = '$U/v_{th}$'

            # flux plot title
            # title = '$\\phi_{ss} / \\phi_{Lawson}$'
            # title = '$\ln \\left( \\phi_{ss} / \\phi_{Lawson} \\right )$'
            # title = '$\log_{10} \\left( \\phi_{ss} / \\phi_{Lawson} \\right )$'
            title = '$\log_{10} \\left( \\phi_{ss} / \\phi_{0} \\right )$'
            title += ', scat_fac=' + str(scat_factor)
            title += ', scat_asym=' + str(scat_asym_factor)
            title += ', ' + str(gas_name)
            title += ', N=' + str(num_cells)

            # Z = np.log(flux_mat * cross_section_main_cell / flux_lawson_ignition_origial)
            # Z = np.log(flux_mat * cross_section_main_cell / flux_lawson_piel)
            # Z = np.log10(flux_mat * cross_section_main_cell / flux_lawson_ignition_piel)
            Z = np.log10(flux_mat / flux_single_naive)
            Z = Z.T

            vmin, vmax = None, None
            # vmin, vmax = 0.5, 2.2
            # vmin, vmax = -3.3, -1.5 # for N=80
            c = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xlabel(x_label, fontsize=axes_label_size)
            ax.set_ylabel(y_label, fontsize=axes_label_size)
            # ax.set_title(title, fontsize=title_fontsize)
            # ax.set_title(gas_name_short, fontsize=title_fontsize)
            fig.suptitle(title, fontsize=title_fontsize)
            fig.colorbar(c, ax=ax)
            fig.set_layout_engine(layout='tight')
            update_format_coord(X, Y, Z, ax=ax)

            # ## plot density profiles
            # fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
            # for ind_Rm, Rm in enumerate(Rm_list):
            # # for ind_Rm, Rm in enumerate([Rm_list[5]]):
            # #     for ind_U, U in enumerate(U_list):
            #     for ind_U, U in enumerate([U_list[2]]):
            #         # print('Rm=', Rm, 'U=', U)
            #         n = mat_dict['n_tR'][ind_Rm, ind_U, :] + mat_dict['n_tL'][ind_Rm, ind_U, :] + mat_dict['n_c'][ind_Rm, ind_U, :]
            #         ax2.plot(n, label='Rm='+ str(Rm) + ',U=' + str(U))
            # ax2.set_xlabel('# cell', fontsize=axes_label_size)
            # ax2.set_ylabel('n [$m^{-3}$]', fontsize=axes_label_size)
            # fig2.suptitle(compiled_set_name)
            # ax2.legend()

            # ### saving figures
            # fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2025/pics/'
            # file_name = 'MMM_flux_' + gas_name
            # file_name += '_scat_fac_' + str(scat_factor)
            # file_name += '_scat_asym_' + str(scat_asym_factor)
            # fig.savefig(fig_save_dir + file_name + '.pdf', format='pdf', dpi=600)
