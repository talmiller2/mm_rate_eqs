import pickle
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib import cm

# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.size': 12})

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
main_dir += 'set56_MM_Rm_10_ni_1e21_Ti_10keV_smooth_fluxeps1e-3'
# main_dir += 'set56_MM_Rm_10_ni_1e21_Ti_10keV_smooth_zeroRL_fluxeps1e-3'


# load single_particle compiled mat
single_particle_dir = '/Users/talmiller/Downloads/single_particle/'
# single_particle_dir = '/home/talm/code/single_particle/slurm_runs/'
single_particle_dir += '/set56_B0_1T_l_1m_Post_Rm_10_intervals_D_T/'

# extract variables from saved single particle calcs
settings_file = single_particle_dir + 'settings.pickle'
with open(settings_file, 'rb') as fid:
    settings_single_particle = pickle.load(fid)
l = settings_single_particle['l']

field_dict_file = single_particle_dir + 'field_dict.pickle'
with open(field_dict_file, 'rb') as fid:
    field_dict_single_particle = pickle.load(fid)
Rm = field_dict_single_particle['Rm']


# num_cells = 10
# num_cells = 30
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

RF_type_list += ['magnetic_transverse']
RF_amplitude_list += [0.04]  # T
induced_fields_factor_list += [1]
with_kr_correction_list += [True]

# RF_type_list += ['magnetic_transverse']
# RF_amplitude_list += [0.04]  # T
# induced_fields_factor_list += [0]
# with_kr_correction_list += [True]

for RF_type, RF_amplitude, induced_fields_factor, with_kr_correction \
        in zip(RF_type_list, RF_amplitude_list, induced_fields_factor_list, with_kr_correction_list):

    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    # fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    for ind_gas, gas_name in enumerate(gas_name_list):
        # for gas_name in gas_name_list:
        # ax = axes[ind_gas]
        ax = axes[0, ind_gas]
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

        time_step_tau_cyclotron_divisions = 50
        # time_step_tau_cyclotron_divisions = 100
        # sigma_r0 = 0
        sigma_r0 = 0.05
        # sigma_r0 = 0.1
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

        # load data of flux for all alpha,beta
        compiled_save_file = main_dir + '/' + set_name + '_N_' + str(num_cells) + '.mat'

        mat_dict = loadmat(compiled_save_file)
        flux_mat = mat_dict['flux_mat']
        alpha_loop_list = mat_dict['alpha_loop_list']
        beta_loop_list = mat_dict['beta_loop_list']

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

        # define the y axis for the 2d plots
        y_array = alpha_loop_list * omega_cyclotron_DTmix / omega_cyclotron_T
        # y_tick_labels = ['{:.2f}'.format(w) for w in y_array]

        # X, Y = np.meshgrid(beta_loop_list, alpha_loop_list)
        # y_label = '$\\alpha$'
        # x_label = '$\\beta$'
        X, Y = np.meshgrid(beta_loop_list, y_array)
        x_label = '$k/\\left( 2 \\pi m^{-1} \\right)$'
        y_label = '$\\omega / \\omega_{0,T}$'

        # flux plot title
        # title = '$\\phi_{ss} / \\phi_{Lawson}$'
        # title = '$\ln \\left( \\phi_{ss} / \\phi_{Lawson} \\right )$'
        # title = '$\log_{10} \\left( \\phi_{ss} / \\phi_{Lawson} \\right )$'
        title = '$\log_{10} \\left( \\phi_{ss} / \\phi_{0} \\right )$'
        title += ' (' + gas_name_short + ')'
        # title += ', ' + RF_type_short + ' ' + RF_amplitude_suffix

        # title += ', ' + RF_type_short + '=' + RF_amplitude_suffix
        # title += ', iff=' + str(induced_fields_factor)
        # title += ', krcor=' + str(with_kr_correction)
        # title += ', N=' + str(num_cells)
        suptitle = RF_type_short + '=' + RF_amplitude_suffix
        suptitle += ', iff=' + str(induced_fields_factor)
        suptitle += ', krcor=' + str(with_kr_correction)
        suptitle += ', N=' + str(num_cells)

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
        ax.set_title(title, fontsize=title_fontsize)
        # ax.set_title(gas_name_short, fontsize=title_fontsize)
        fig.suptitle(suptitle, fontsize=title_fontsize)
        fig.colorbar(c, ax=ax)
        fig.set_layout_engine(layout='tight')
        update_format_coord(X, Y, Z, ax=ax)

        ### take the density solution for each case, and calculate the power based on the E_ratio
        time_step_tau_cyclotron_divisions = 50
        sigma_r0 = 0.05
        radial_distribution = 'uniform'
        theta_type = 'sign_vz'
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
        # print(set_name)
        single_particle_file = single_particle_dir + '/' + set_name + '.mat'
        single_particle_mat_dict = loadmat(single_particle_file)

        # estimate the RF power in the plug
        E_ini_per_particle = settings_single_particle['kB_eV'] * settings_single_particle['T_keV'] * 1e3  # [Joule]
        A = settings['cross_section_main_cell']
        cell_volume = l * A

        # # simplest model assuming const density and known volume 1m^3
        # N_particles = state['n'][0] * 1 # density 1e21[m^-3] in volume 1[m^3]
        # E_ini_total = E_ini_per_particle * N_particles  # [Joule]
        # E_fin_total = E_ini_total * single_particle_mat_dict['E_ratio']
        # power_total_W = (E_fin_total - E_ini_total) / settings_single_particle['t_max'] # [Watt=Joule/s]

        # calc based on changing density in plug cells and different power per population
        power_W_dict = {}
        power_total_W = 0
        for pop in ['c', 'tR', 'tL']:
            # for pop in ['tR']:
            # for pop in ['tL']:
            # for pop in ['c']:

            if pop == 'tR':
                pop_single_particle = 'R'
            elif pop == 'tL':
                # pop_single_particle = 'L'
                pop_single_particle = 'C'  # because of mistake in mixing L-C in the single particle compilation
            else:
                # pop_single_particle = 'C'
                pop_single_particle = 'L'  # because of mistake in mixing L-C in the single particle compilation
            E_ratio = single_particle_mat_dict['E_ratio_' + pop_single_particle]

            power_W_dict[pop] = 0

            # for ind_cell in range(len(state['n'])):
            for ind_cell in range(mat_dict['n'].shape[2]):
                # for ind_cell in [0]:
                n_curr = mat_dict['n_' + pop][:, :, ind_cell]
                power_per_particle = E_ini_per_particle * (E_ratio - 1) / settings_single_particle[
                    't_max']  # [Watt=Joule/s]
                power_W_dict[pop] += cell_volume * n_curr * power_per_particle

            power_total_W += power_W_dict[pop]

        power_total_MW = power_total_W / 1e6
        Z = power_total_MW
        # title = 'Power [MW]'
        # title += ', ' + RF_type_short + '=' + RF_amplitude_suffix
        # title += ', iff=' + str(induced_fields_factor)
        # title += ', krcor=' + str(with_kr_correction)
        # title += ', N=' + str(num_cells)
        # title_power = 'Power [MW]'
        title_power = 'Power [MW] (' + gas_name_short + ')'
        # ax = axes2[ind_gas]
        ax = axes[1, ind_gas]
        Z = power_total_MW
        Z = Z.T
        vmin, vmax = None, None
        # vmin, vmax = 0, 1000 # for N=80
        c = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel(x_label, fontsize=axes_label_size)
        ax.set_ylabel(y_label, fontsize=axes_label_size)
        # ax.set_title(title, fontsize=title_fontsize)
        # ax.set_title(gas_name_short, fontsize=title_fontsize)
        ax.set_title(title_power, fontsize=title_fontsize)
        # fig.suptitle(title, fontsize=title_fontsize)
        fig.colorbar(c, ax=ax)
        fig.set_layout_engine(layout='tight')
        update_format_coord(X, Y, Z, ax=ax)
