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

plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 3

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs//runs/slurm_runs/'
main_dir += 'set62_MMM_ni_1e21_Ti_10keV_constmfp'

# linewidth = 1
linewidth = 2

# cmap = 'viridis'
# cmap = 'plasma'
# cmap = 'inferno'
cmap = 'coolwarm'

num_cells_list = [20, 40]

modes = []
modes += ['mahmir']
modes += ['mekel']

ft_list = []
ft_list += [1]
ft_list += [2]

for num_cells in num_cells_list:
    for mode in modes:
        for ft in ft_list:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))

            compiled_set_name = 'compiled_'
            compiled_set_name += 'N_' + str(num_cells)
            compiled_set_name += '_mode_' + str(mode)
            compiled_set_name += f'_ft_{ft:g}'
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
            title += ', mode=' + str(mode)
            title += ', ft=' + str(ft)
            title += ', N=' + str(num_cells)

            # # change failed nan to zero
            # flux_mat = np.nan_to_num(flux_mat, nan=np.nanmin(flux_mat))

            # Z = np.log(flux_mat * cross_section_main_cell / flux_lawson_ignition_origial)
            # Z = np.log(flux_mat * cross_section_main_cell / flux_lawson_piel)
            # Z = np.log10(flux_mat * cross_section_main_cell / flux_lawson_ignition_piel)
            Z = np.log10(flux_mat / flux_single_naive)
            Z = Z.T

            vmin, vmax = None, None
            # vmin, vmax = 0.5, 2.2
            # vmin, vmax = -3.3, -1.5 # for N=80
            c = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            # ax.set_title(title, fontsize=title_fontsize)
            # ax.set_title(gas_name_short, fontsize=title_fontsize)
            # fig.suptitle(title)
            fig.colorbar(c, ax=ax)
            fig.set_layout_engine(layout='tight')
            update_format_coord(X, Y, Z, ax=ax)

            # ### saving figures
            # fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2026/pics/'
            # file_name = f'MMM_flux_of_Rm_U_at_fixed_N_{num_cells}'
            # # file_name += f'_mode_{mode}'
            # if mode == 'mahmir':
            #     A = 0
            # elif mode == 'mekel':
            #     A = 1
            # file_name += f'_A_{A}'
            # file_name += f'_ft_{ft:g}'
            # fig.savefig(fig_save_dir + file_name + '.pdf', format='pdf', dpi=600)
