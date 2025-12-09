import pickle
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib import cm

# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 12})

# axes_label_size = 12
axes_label_size = 14
# axes_label_size = 18
# title_fontsize = 12
title_fontsize = 14

import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from mm_rate_eqs.fusion_functions import get_lawson_parameters, get_lawson_criterion_piel

from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss, get_magnetic_pressure, \
    get_ideal_gas_pressure, get_ideal_gas_energy_per_volume, get_magnetic_field_for_given_pressure, \
    get_bohm_diffusion_constant, get_larmor_radius

from mm_rate_eqs.plot_functions import update_format_coord

from mm_rate_eqs.plasma_functions import define_plasma_parameters, get_larmor_frequency

B0 = 1  # [T]
omega_cyclotron_DTmix = 2 * np.pi * get_larmor_frequency(B0, gas_name='DT_mix')
omega_cyclotron_T = 2 * np.pi * get_larmor_frequency(B0, gas_name='tritium')

plt.close('all')

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs//runs/slurm_runs/'
# main_dir += 'set47_MM_Rm_10_ni_1e21_Ti_10keV_withRMF'
# main_dir += 'set48_MM_Rm_10_ni_1e21_Ti_10keV_withRMF_zeroRL_fluxeps1e-2'
# main_dir += 'set49_MM_Rm_10_ni_1e21_Ti_10keV_withRMF_fluxeps1e-2'
# main_dir += 'set50_MM_Rm_10_ni_1e20_Ti_10keV_withRMF_zeroRL_fluxeps1e-2'
# main_dir += 'set54_MM_Rm_10_ni_1e21_Ti_10keV_smooth_fluxeps1e-3'
# main_dir += 'set55_MM_Rm_10_ni_1e21_Ti_10keV_smooth_fluxeps1e-3'
# main_dir += 'set56_MM_Rm_10_ni_1e21_Ti_10keV_smooth_fluxeps1e-3'
# main_dir += 'set56_MM_Rm_10_ni_1e21_Ti_10keV_smooth_zeroRL_fluxeps1e-3'
# main_dir += '/set59_MMwithRF_Rm_5_ni_1e21_Ti_10keV_smooth/'
main_dir += '/set61_MMwithRF_Rm_5_ni_1e21_Ti_10keV_smooth/'

# load single_particle compiled mat
single_particle_dir = '/Users/talmiller/Downloads/single_particle/'
# single_particle_dir = '/home/talm/code/single_particle/slurm_runs/'
# single_particle_dir += '/set56_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'
# single_particle_dir += '/set59_B0_1T_l_1m_Post_Rm_5_r0max_30cm/'
single_particle_dir += '/set61_B0_1T_l_1m_Post_Rm_5_r0max_10cm/'

# extract variables from saved single particle calcs
settings_file = single_particle_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings_single_particle = pickle.load(fid)
l = settings_single_particle['l']

field_dict_file = single_particle_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict_single_particle = pickle.load(fid)
Rm = field_dict_single_particle['Rm']

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

# _, flux_lawson_ignition_origial = get_lawson_parameters(ni, Ti_keV, settings)
# _, flux_lawson_piel, _, flux_lawson_ignition_piel = get_lawson_criterion_piel(ni, Ti_keV, settings)
cross_section_main_cell = settings['cross_section_main_cell']
v_th = state['v_th'][0]
flux_single_naive = ni * v_th

# num_cells = 10
# num_cells = 30
# num_cells = 50
# num_cells = 80
num_cells_list = [10, 20, 30, 40, 50, 60, 70, 80]

# linewidth = 1
linewidth = 2

# cmap = 'viridis'
# cmap = 'plasma'
# cmap = 'inferno'
cmap = 'coolwarm'

save_figures = False
# save_figures = True

gas_name_list = ['deuterium', 'tritium']
# gas_name_list = ['tritium']

RF_type_list = []
RF_amplitude_list = []
induced_fields_factor_list = []
with_kr_correction_list = []

# RF_type_list += ['electric_transverse']
# RF_amplitude_list += [25]  # kV/m
# induced_fields_factor_list += [1]
# with_kr_correction_list += [True]

RF_type_list += ['electric_transverse']
RF_amplitude_list += [50]  # kV/m
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

# RF_type_list += ['magnetic_transverse']
# RF_amplitude_list += [0.02]  # T
# induced_fields_factor_list += [1]
# with_kr_correction_list += [True]
#
# RF_type_list += ['magnetic_transverse']
# RF_amplitude_list += [0.02]  # T
# induced_fields_factor_list += [0]
# with_kr_correction_list += [True]

# RF_type_list += ['magnetic_transverse']
# RF_amplitude_list += [0.04]  # T
# induced_fields_factor_list += [1]
# with_kr_correction_list += [True]
#
# RF_type_list += ['magnetic_transverse']
# RF_amplitude_list += [0.04]  # T
# induced_fields_factor_list += [0]
# with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.05]  # T
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.05]  # T
induced_fields_factor_list += [0]
with_kr_correction_list += [True]

# RF_type_list += ['magnetic_transverse']
# RF_amplitude_list += [0.025]  # T
# induced_fields_factor_list += [1]
# with_kr_correction_list += [True]
#
# RF_type_list += ['magnetic_transverse']
# RF_amplitude_list += [0.025]  # T
# induced_fields_factor_list += [0]
# with_kr_correction_list += [True]

# ind_alpha_list = [2, 10, 2]
# ind_beta_list = [2, 10, 15]
# ind_alpha_list = [0, 10, 20]
# ind_beta_list = [0, 10, 20]
# ind_alpha_list = [0, 10,  10, 20,  20]
# ind_beta_list =  [0, 10, -10, 20, -20]
ind_alpha_list = [10, 13, 13]
ind_beta_list = [10, 0, 20]
color_list = ['b', 'g', 'r', 'orange', 'k']
linestyle_list = ['-', '--', ':', '-.']

for ind_gas, gas_name in enumerate(gas_name_list):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    if gas_name == 'deuterium':
        gas_name_short = 'D'
    else:
        gas_name_short = 'T'

    for RF_type, RF_amplitude, induced_fields_factor, with_kr_correction, linestyle \
            in zip(RF_type_list, RF_amplitude_list, induced_fields_factor_list, with_kr_correction_list,
                   linestyle_list):

        if RF_type == 'magnetic_transverse':
            RF_type_short = 'RMF'
            RF_amplitude_suffix = str(int(1e3 * RF_amplitude)) + '[mT]'
        else:
            RF_type_short = 'REF'
            RF_amplitude_suffix = str(int(RF_amplitude)) + '[kV/m]'

        time_step_tau_cyclotron_divisions = 50
        # time_step_tau_cyclotron_divisions = 100
        # sigma_r0 = 0
        # sigma_r0 = 0.05
        sigma_r0 = 0.1
        # sigma_r0 = 0.3
        radial_distribution = 'uniform'

        # theta_type = 'sign_vz0'
        theta_type = 'sign_vz'

        # set_name = 'compiled_'
        set_name = 'smooth_compiled_'
        set_name += theta_type + '_'
        if RF_type == 'electric_transverse':
            set_name += 'ERF_' + str(RF_amplitude)
        elif RF_type == 'magnetic_transverse':
            set_name += 'BRF_' + str(RF_amplitude)
        if induced_fields_factor < 1.0:
            set_name += '_iff' + str(induced_fields_factor)
        if with_kr_correction == True:
            set_name += '_withkrcor'
        set_name += '_tcycdivs' + str(time_step_tau_cyclotron_divisions)
        if sigma_r0 > 0:
            set_name += '_sigmar' + str(sigma_r0)
            if radial_distribution == 'normal':
                set_name += 'norm'
            elif radial_distribution == 'uniform':
                set_name += 'unif'
        set_name += '_' + gas_name
        print(set_name)

        for ind_alpha, ind_beta, color in zip(ind_alpha_list, ind_beta_list, color_list):
            phi_list = []
            for ind_num_cells, num_cells in enumerate(num_cells_list):
                # load data of flux for all alpha,beta
                compiled_save_file = main_dir + '/' + set_name + '_N_' + str(num_cells) + '.mat'

                mat_dict = loadmat(compiled_save_file)
                flux_mat = mat_dict['flux_mat']
                alpha_loop_list = mat_dict['alpha_loop_list'][0]
                beta_loop_list = mat_dict['beta_loop_list'][0]

                flux_curr = flux_mat[ind_beta, ind_alpha]
                phi = flux_curr / flux_single_naive
                phi_list += [phi]

            alpha_curr = alpha_loop_list[ind_alpha]
            beta_curr = beta_loop_list[ind_beta]
            # label = '$\\omega / \\omega_{0,T}$=' + str(alpha_curr) + ', $k/\\left( 2 \\pi m^{-1} \\right)$=' + str(beta_curr)
            label = f"$\\omega / \\omega_{{0,T}}$={alpha_curr:.2g}, $k/\\left( 2 \\pi m^{{-1}} \\right)$={beta_curr:g}"
            if linestyle != '-': label = None
            ax.plot(num_cells_list, phi_list,
                    marker='o',
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    )

    x_label = 'number of cells'
    y_label = '$\\phi_{ss} / \\phi_{0}$'
    title = 'steady state flux: ' + gas_name
    plt.xlabel(x_label, fontsize=axes_label_size)
    plt.ylabel(y_label, fontsize=axes_label_size)
    plt.title(title, fontsize=title_fontsize)
    plt.yscale('log')
    legend_data = plt.legend(fontsize=axes_label_size)
    plt.grid(True)
    plt.tight_layout()

    # adding "fake legend" to explain the linestyles
    from matplotlib.lines import Line2D

    linestyle_key = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=2),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2),
        Line2D([0], [0], color='black', linestyle=':', linewidth=2),
    ]

    legend_style = plt.legend(handles=linestyle_key,
                              labels=['TREF', 'TRMF', 'TRMF w/o E'],
                              # loc='lower left',  # choose a spot that doesn’t overlap
                              loc='upper right',  # choose a spot that doesn’t overlap
                              # title='Linestyle meaning'
                              )

    # add the first legend back (this is the crucial line!)
    plt.gca().add_artist(legend_data)

    # ### saving figures
    # fig_save_dir = '/Users/talmiller/Data/UNI/Courses Graduate/Plasma/Papers/texts/paper_2025/pics/'
    # file_name = 'compiled_flux_of_N'
    # file_suffix = ''
    # # if RF_type == 'electric_transverse':
    # #     file_suffix += '_REF'
    # # else:
    # #     file_suffix += '_RMF'
    # # if induced_fields_factor < 1.0: file_suffix += '_iff' + str(induced_fields_factor)
    # file_suffix += '_' + gas_name_short
    # fig.savefig(fig_save_dir + file_name + file_suffix + '.pdf', format='pdf', dpi=600)
