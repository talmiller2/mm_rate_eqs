import matplotlib
# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib import cm

# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 12})

import numpy as np
from scipy.optimize import curve_fit

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
main_dir += 'set49_MM_Rm_10_ni_1e21_Ti_10keV_withRMF_fluxeps1e-2'
# main_dir += 'set50_MM_Rm_10_ni_1e20_Ti_10keV_withRMF_zeroRL_fluxeps1e-2'

num_cells = 50
# num_cells = 80

# linewidth = 1
linewidth = 2

axes_label_size = 12
title_fontsize = 12

# cmap = 'viridis'
# cmap = 'plasma'
# cmap = 'inferno'
cmap = 'coolwarm'

gas_name_list = ['deuterium', 'tritium']
# gas_name_list = ['tritium']

RF_type_list = []
RF_amplitude_list = []
induced_fields_factor_list = []
with_kr_correction_list = []

RF_type_list += ['electric_transverse']
RF_amplitude_list += [25]  # kV/m
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['electric_transverse']
RF_amplitude_list += [50]  # kV/m
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.02]  # T
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.02]  # T
induced_fields_factor_list += [0]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.04]  # T
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.04]  # T
induced_fields_factor_list += [0]
with_kr_correction_list += [True]

for RF_type, RF_amplitude, induced_fields_factor, with_kr_correction \
        in zip(RF_type_list, RF_amplitude_list, induced_fields_factor_list, with_kr_correction_list):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for gas_name, ax in zip(gas_name_list, axes):
        # for gas_name in gas_name_list:

        if gas_name == 'deuterium':
            gas_name_short = 'D'
        else:
            gas_name_short = 'T'

        if RF_type == 'magnetic_transverse':
            RF_type_short = 'RMF'
            RF_amplitude_suffix = str(int(1e3 * RF_amplitude)) + '[mT]'
        else:
            RF_type_short = 'REF'
            RF_amplitude_suffix = str(int(RF_amplitude)) + '[kV/m]'

        # title = '$\\phi_{ss} / \\phi_{Lawson}$'
        # title = '$\ln \\left( \\phi_{ss} / \\phi_{Lawson} \\right )$'
        title = '$\log_{10} \\left( \\phi_{ss} / \\phi_{0} \\right )$'
        # title = gas_name_short
        # title += ', ' + RF_type_short + ' ' + RF_amplitude_suffix
        title += ', ' + RF_type_short + '=' + RF_amplitude_suffix
        title += ', iff=' + str(induced_fields_factor)
        title += ', krcor=' + str(with_kr_correction)
        title += ', N=' + str(num_cells)

        time_step_tau_cyclotron_divisions = 50
        # time_step_tau_cyclotron_divisions = 100
        # sigma_r0 = 0
        sigma_r0 = 0.05
        # sigma_r0 = 0.1
        radial_distribution = 'uniform'

        # theta_type = 'sign_vz0'
        theta_type = 'sign_vz'

        set_name = 'compiled_'
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

        # load data of flux for all alpha,beta
        compiled_save_file = main_dir + '/' + set_name + '_N_' + str(num_cells) + '.mat'

        mat_dict = loadmat(compiled_save_file)
        flux_mat = mat_dict['flux_mat']
        alpha_loop_list = mat_dict['alpha_loop_list']
        beta_loop_list = mat_dict['beta_loop_list']

        # load on of the settings files
        # mm_rate_eqs_sim_dir = main_dir + '/' + set_name + '/'
        # import glob
        state_file = main_dir + '/state.pickle'
        settings_file = main_dir + '/settings.pickle'
        state, settings = load_simulation(state_file, settings_file)

        # post process the flux normalization
        ni = state['n'][0]
        Ti_keV = state['Ti'][0] / 1e3
        _, flux_lawson_piel, _, flux_lawson_ignition_piel = get_lawson_criterion_piel(ni, Ti_keV, settings)
        cross_section_main_cell = settings['cross_section_main_cell']
        v_th = state['v_th'][0]
        flux_single_naive = ni * v_th

        # define the y axis for the 2d plots
        y_array = alpha_loop_list * omega_cyclotron_DTmix / omega_cyclotron_T
        # y_tick_labels = ['{:.2f}'.format(w) for w in y_array]

        # X, Y = np.meshgrid(beta_loop_list, alpha_loop_list)
        # y_label = '$\\alpha$'
        # x_label = '$\\beta$'
        X, Y = np.meshgrid(beta_loop_list, y_array)
        x_label = '$k/\\left( 2 \\pi m^{-1} \\right)$'
        y_label = '$\\omega / \\omega_{0,T}$'

        # Z = np.log(flux_mat * cross_section_main_cell / flux_lawson_ignition_piel)
        Z = np.log10(flux_mat / flux_single_naive)
        Z = Z.T

        vmin, vmax = None, None
        c = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel(x_label, fontsize=axes_label_size)
        ax.set_ylabel(y_label, fontsize=axes_label_size)
        # ax.set_title(title, fontsize=title_fontsize)
        ax.set_title(gas_name_short, fontsize=title_fontsize)
        fig.suptitle(title, fontsize=title_fontsize)
        fig.colorbar(c, ax=ax)
        fig.set_layout_engine(layout='tight')
        update_format_coord(X, Y, Z, ax=ax)
