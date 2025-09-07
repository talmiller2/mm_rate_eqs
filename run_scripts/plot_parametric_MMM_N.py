import matplotlib
# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from mm_rate_eqs.fusion_functions import get_lawson_parameters, get_lawson_criterion_piel
from mm_rate_eqs.plot_functions import update_format_coord

plt.close('all')

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs//runs/slurm_runs/'
# main_dir += 'set57_MMM_ni_1e21_Ti_10keV_constmfp'
main_dir += 'set58_MMM_ni_1e21_Ti_10keV_constmfp_trfix'

# linewidth = 1
linewidth = 2

axes_label_size = 12
# axes_label_size = 18
title_fontsize = 12

# cmap = 'viridis'
# cmap = 'plasma'
# cmap = 'inferno'
cmap = 'coolwarm'

# gas_name = 'deuterium'
gas_name = 'tritium'

# num_cells_list = [10]
# num_cells_list = [30]
# num_cells_list = [50]
# num_cells_list = [10, 30, 50, 80]
num_cells_list = [10, 20, 30, 40, 50, 60, 70, 80]

# scat_factor_list = [0.1, 1]
# scat_factor_list = [0.1]
scat_factor_list = [1]

# scat_asym_list = [0.5, 1, 2]
# scat_asym_list = [0.5, 1]
# scat_asym_list = [0.5]
scat_asym_list = [1]
# scat_asym_list = [2]

inds_Rm = [2, 6]
inds_U = [2, 4, 6]
# inds_Rm = [5, 10]
# inds_U = [5, 10]

color_list = ['b', 'g', 'r']
linestyle_list = ['-', '--', ':']

for scat_factor in scat_factor_list:
    for scat_asym_factor in scat_asym_list:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        phi = {}
        for num_cells in num_cells_list:
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

            for ind_U in inds_U:
                U = U_list[ind_U]
                for ind_Rm in inds_Rm:
                    Rm = Rm_list[ind_Rm]

                    key = f'Rm_{Rm}_U_{U}'
                    if key not in phi:
                        phi[key] = []

                    phi[key] += [flux_mat[ind_Rm, ind_U] / flux_single_naive]

        # plot phi(N)
        for ind_U, color in zip(inds_U, color_list):
            U = U_list[ind_U]
            for ind_Rm, linestyle in zip(inds_Rm, linestyle_list):
                Rm = Rm_list[ind_Rm]

                key = f'Rm_{Rm}_U_{U}'
                label = f'$U/v_{{th}}$={U}, $R_m$={Rm}'
                x_label = 'number of cells'
                y_label = '$\\phi_{ss} / \\phi_{0}$'
                plt.plot(num_cells_list, phi[key], label=label, marker='o', color=color, linestyle=linestyle)

                plt.xlabel(x_label, fontsize=axes_label_size)
                plt.ylabel(y_label, fontsize=axes_label_size)
                plt.yscale('log')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                # ### saving figures
                # fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2026/pics/'
                # file_name = 'MMM_flux_of_N_' + gas_name
                # file_name += '_scat_fac_' + str(scat_factor)
                # file_name += '_scat_asym_' + str(scat_asym_factor)
                # # file_name += '_novcol'
                # fig.savefig(fig_save_dir + file_name + '.pdf', format='pdf', dpi=600)
