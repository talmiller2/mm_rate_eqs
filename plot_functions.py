import logging
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_relaxation_status(state, settings):
    plt.rcParams.update({'font.size': 16})

    if state['termination_criterion_reached'] is True:
        linewidth = 3
        linestyle = '-'
    else:
        linewidth = 1
        linestyle = '--'
    label_suffix = ''

    # define x axis
    if settings['plots_x_axis'] == 'total_length':
        z_array = np.cumsum(state['mirror_cell_sizes'])
    elif settings['plots_x_axis'] == 'cell_number':
        z_array = np.linspace(1, settings['number_of_cells'], settings['number_of_cells'])
    else:
        raise ValueError('invalid plots_x_axis: ' + settings['plots_x_axis'])

    plt.figure(1)
    plt.plot(z_array, state['n_tR'], linewidth=linewidth, linestyle=linestyle, color='b', label='n_tR' + label_suffix)
    plt.plot(z_array, state['n_tL'], linewidth=linewidth, linestyle=linestyle, color='g', label='n_tL' + label_suffix)
    plt.plot(z_array, state['n_c'], linewidth=linewidth, linestyle=linestyle, color='r', label='n_c' + label_suffix)
    plt.plot(z_array, state['n'], linewidth=linewidth, linestyle=linestyle, color='k', label='n' + label_suffix)

    plt.figure(2)
    plt.plot(z_array, state['flux_trans_R'], linestyle=linestyle, label='flux_trans_R' + label_suffix,
             linewidth=linewidth, color='b')
    plt.plot(z_array, state['flux_trans_L'], linestyle=linestyle, label='flux_trans_L' + label_suffix,
             linewidth=linewidth, color='g')
    plt.plot(z_array, state['flux_mmm_drag'], linestyle=linestyle, label='flux_mmm_drag' + label_suffix,
             linewidth=linewidth, color='r')
    plt.plot(z_array, state['flux'], linestyle=linestyle, label='flux total' + label_suffix, linewidth=linewidth,
             color='k')

    plt.figure(3)
    plt.plot(z_array, state['coulomb_scattering_rate'], linestyle=linestyle, label='scattering rate' + label_suffix,
             linewidth=linewidth, color='b')
    plt.plot(z_array, state['transmission_rate'], linestyle=linestyle, label='transmission rate' + label_suffix,
             linewidth=linewidth, color='g')
    plt.plot(z_array, state['mmm_drag_rate'], linestyle=linestyle, label='MMM drag rate' + label_suffix,
             linewidth=linewidth, color='r')

    plt.figure(4)
    plt.plot(z_array, state['Ti'] / settings['keV'], linewidth=linewidth, linestyle=linestyle, color='r',
             label='Ti' + label_suffix)
    plt.plot(z_array, state['Te'] / settings['keV'], linewidth=linewidth, linestyle=linestyle, color='b',
             label='Te' + label_suffix)

    plt.figure(5)
    plt.plot(z_array, state['mean_free_path'], linestyle=linestyle, label='mfp' + label_suffix, linewidth=linewidth,
             color='r')

    plt.figure(6)
    plt.plot(z_array, state['alpha_tR'], linestyle=linestyle, label='$\\alpha_{tR}$' + label_suffix,
             linewidth=linewidth, color='b')
    plt.plot(z_array, state['alpha_tL'], linestyle=linestyle, label='$\\alpha_{tL}$' + label_suffix,
             linewidth=linewidth, color='g')
    plt.plot(z_array, state['alpha_c'], linestyle=linestyle, label='$\\alpha_{c}$' + label_suffix,
             linewidth=linewidth, color='r')

    plt.figure(7)
    plt.plot(state['mirror_cell_sizes'], linestyle=linestyle, linewidth=linewidth, color='b')

    plt.figure(8)
    plt.plot(state['mean_free_path'] / state['mirror_cell_sizes'], linestyle=linestyle, linewidth=linewidth, color='b')

    plt.figure(9)
    plt.plot(z_array, state['v_th'], linestyle=linestyle, linewidth=linewidth, color='b',
             label='$v_{th}$' + label_suffix)
    plt.plot(z_array, state['U'], linestyle=linestyle, linewidth=linewidth, color='r',
             label='$U_{MMM}$' + label_suffix)
    plt.plot(z_array, state['v_col'], linestyle=linestyle, linewidth=linewidth, color='c',
             label='$v_{col}$' + label_suffix)

    plt.figure(10)
    plt.plot(state['t_evolution'], state['flux_normalized_std_evolution'], linestyle=linestyle, linewidth=linewidth,
             label='normalized flux std' + label_suffix, color='k')

    plt.figure(11)
    plt.plot(state['t_evolution'], state['flux_min_evolution'], linestyle=linestyle, linewidth=linewidth,
             label='flux max' + label_suffix, color='r')
    plt.plot(state['t_evolution'], state['flux_max_evolution'], linestyle=linestyle, linewidth=linewidth,
             label='flux min' + label_suffix, color='b')

    plt.figure(12)
    plt.plot(z_array, state['n_c'] / state['n_tR'], linewidth=linewidth, linestyle=linestyle, color='r',
             label='n_c/n_tR' + label_suffix)
    plt.plot(z_array, state['n_tL'] / state['n_tR'], linewidth=linewidth, linestyle=linestyle, color='g',
             label='n_tL/n_tR' + label_suffix)

    plt.figure(13)
    inds = np.where(state['alpha_tR'] > 0)
    plt.plot(z_array[inds], state['n_tR'][inds] / state['n'][inds] / state['alpha_tR'][inds], linestyle=linestyle,
             label='$n/\\alpha$ tR fraction' + label_suffix,
             linewidth=linewidth, color='b')
    inds = np.where(state['alpha_tL'] > 0)
    plt.plot(z_array[inds], state['n_tL'][inds] / state['n'][inds] / state['alpha_tL'][inds], linestyle=linestyle,
             label='$n/\\alpha$ tL fraction' + label_suffix,
             linewidth=linewidth, color='g')
    inds = np.where(state['alpha_c'] > 0)
    plt.plot(z_array[inds], state['n_c'][inds] / state['n'][inds] / state['alpha_c'][inds], linestyle=linestyle,
             label='$n/\\alpha$ c fraction' + label_suffix,
             linewidth=linewidth, color='r')

    plt.figure(14)
    plt.plot(z_array, state['flux_E'], linestyle=linestyle, label='flux_E' + label_suffix,
             linewidth=linewidth, color='b')

    if settings['save_plots'] is True:
        plot_relaxation_end(settings, save_plots=True)

    return


def plot_relaxation_end(settings, title_name='', show_legend=False, save_plots=False):
    # define x axis
    if settings['plots_x_axis'] == 'total_length':
        xlabel = 'z [m]'
    elif settings['plots_x_axis'] == 'cell_number':
        xlabel = 'cell number'
    else:
        raise ValueError('invalid plots_x_axis: ' + settings['plots_x_axis'])

    plt.figure(1)
    plt.ylabel('$m^{-3}$')
    plt.xlabel(xlabel)

    plt.figure(2)
    plt.ylabel('flux')
    plt.xlabel(xlabel)

    plt.figure(3)
    plt.ylabel('rate [$s^{-1}$]')
    plt.xlabel(xlabel)
    plt.yscale('log')

    plt.figure(4)
    plt.ylabel('T [keV]')
    plt.xlabel(xlabel)

    plt.figure(5)
    plt.ylabel('mfp [m]')
    plt.xlabel(xlabel)

    plt.figure(6)
    plt.ylabel('$\\alpha$')
    plt.xlabel(xlabel)

    plt.figure(7)
    plt.xlabel('cell number')
    plt.ylabel('cell size [m]')

    plt.figure(8)
    plt.xlabel('cell number')
    plt.ylabel('mfp / cell size')

    plt.figure(9)
    plt.ylabel('velocity [m/s]')
    plt.xlabel(xlabel)

    plt.figure(10)
    plt.ylabel('flux normalized std evolution')
    plt.xlabel('simulation time')

    plt.figure(11)
    plt.ylabel('fluxes min/max evolution')
    plt.xlabel('simulation time')

    plt.figure(12)
    plt.ylabel('density ratios')
    plt.xlabel(xlabel)

    plt.figure(13)
    plt.ylabel('$n/n_{tot}/\\alpha$ ratios')
    plt.xlabel(xlabel)
    plt.ylim([0, 2])

    plt.figure(14)
    plt.ylabel('energy flux')
    plt.xlabel(xlabel)

    num_plots = 14
    for fig_num in range(1, num_plots + 1):
        plt.figure(fig_num)
        plt.tight_layout()
        plt.grid(True)
        plt.title(title_name)
        if show_legend is True: plt.legend()

        if save_plots is True:
            # create save directory
            if not os.path.exists(settings['save_dir']):
                logging.info('Creating save directory: ' + settings['save_dir'])
            plt.savefig(settings['save_dir'] + '/fig' + str(fig_num))

    return
