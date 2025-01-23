import logging
import os

import matplotlib.pyplot as plt
import numpy as np

def plot_relaxation_status(state, settings):
    plt.rcParams.update({'font.size': settings['fontsize']})

    if state['termination_criterion_reached'] is True:
        linewidth = 3
        linestyle = '-'
    else:
        linewidth = 1
        linestyle = '--'

    # override to plot in a different style for comparisons
    if 'linestyle' in settings:
        linestyle = settings['linestyle']

    label_suffix = ''

    # define x axis
    if settings['plots_x_axis'] == 'total_length':
        z_array = np.cumsum(state['mirror_cell_sizes'])
        xlabel = 'z [m]'
    elif settings['plots_x_axis'] == 'cell_number':
        z_array = np.linspace(1, settings['number_of_cells'], settings['number_of_cells'])
        xlabel = 'cell number'
    else:
        raise ValueError('invalid plots_x_axis: ' + settings['plots_x_axis'])

    fig_cnt = 0

    fig_cnt += 1
    plt.figure(fig_cnt)
    plt.plot(z_array, state['n_tR'], linewidth=linewidth, linestyle=linestyle, color='b', label='n_tR' + label_suffix)
    plt.plot(z_array, state['n_tL'], linewidth=linewidth, linestyle=linestyle, color='g', label='n_tL' + label_suffix)
    plt.plot(z_array, state['n_c'], linewidth=linewidth, linestyle=linestyle, color='r', label='n_c' + label_suffix)
    plt.plot(z_array, state['n'], linewidth=linewidth, linestyle=linestyle, color='k', label='n' + label_suffix)
    if state['termination_criterion_reached'] is True:
        plt.ylabel('$m^{-3}$')
        plt.xlabel(xlabel)

    fig_cnt += 1
    plt.figure(fig_cnt)
    plt.plot(z_array, state['flux_trans_R'], linestyle=linestyle, label='flux_trans_R' + label_suffix,
             linewidth=linewidth, color='b')
    plt.plot(z_array, state['flux_trans_L'], linestyle=linestyle, label='flux_trans_L' + label_suffix,
             linewidth=linewidth, color='g')
    plt.plot(z_array, state['flux_mmm_drag'], linestyle=linestyle, label='flux_mmm_drag' + label_suffix,
             linewidth=linewidth, color='r')
    plt.plot(z_array, state['flux'], linestyle=linestyle, label='flux total' + label_suffix, linewidth=linewidth,
             color='k')
    if state['termination_criterion_reached'] is True:
        plt.ylabel('flux')
        plt.xlabel(xlabel)

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # plt.plot(z_array, state['coulomb_scattering_rate'], linestyle=linestyle, label='scattering rate' + label_suffix,
    #          linewidth=linewidth, color='k')
    # if 'transmission_rate_R' in state:
    #     plt.plot(z_array, state['transmission_rate_R'], linestyle=linestyle, label='transmission rate R' + label_suffix,
    #              linewidth=linewidth, color='b')
    #     plt.plot(z_array, state['transmission_rate_L'], linestyle=linestyle, label='transmission rate L' + label_suffix,
    #              linewidth=linewidth, color='g')
    # plt.plot(z_array, state['mmm_drag_rate'], linestyle=linestyle, label='MMM drag rate' + label_suffix,
    #          linewidth=linewidth, color='r')
    # if state['termination_criterion_reached'] is True:
    #     plt.ylabel('rate [$s^{-1}$]')
    #     plt.xlabel(xlabel)
    #     plt.yscale('log')

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # plt.plot(z_array, state['Ti'] / settings['keV'], linewidth=linewidth, linestyle=linestyle, color='r',
    #          label='Ti' + label_suffix)
    # plt.plot(z_array, state['Te'] / settings['keV'], linewidth=linewidth, linestyle=linestyle, color='b',
    #          label='Te' + label_suffix)
    # if state['termination_criterion_reached'] is True:
    #     plt.ylabel('T [keV]')
    #     plt.xlabel(xlabel)

    fig_cnt += 1
    plt.figure(fig_cnt)
    plt.plot(z_array, state['mean_free_path'], linestyle=linestyle, label='mfp' + label_suffix, linewidth=linewidth,
             color='r')
    if state['termination_criterion_reached'] is True:
        plt.ylabel('mfp [m]')
        plt.xlabel(xlabel)

    fig_cnt += 1
    plt.figure(fig_cnt)
    plt.plot(z_array, state['alpha_tR'], linestyle=linestyle, label='$\\alpha_{tR}$' + label_suffix,
             linewidth=linewidth, color='b')
    plt.plot(z_array, state['alpha_tL'], linestyle=linestyle, label='$\\alpha_{tL}$' + label_suffix,
             linewidth=linewidth, color='g')
    plt.plot(z_array, state['alpha_c'], linestyle=linestyle, label='$\\alpha_{c}$' + label_suffix,
             linewidth=linewidth, color='r')
    if state['termination_criterion_reached'] is True:
        plt.ylabel('$\\alpha$')
        plt.xlabel(xlabel)

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # plt.plot(state['mirror_cell_sizes'], linestyle=linestyle, linewidth=linewidth, color='b')
    # if state['termination_criterion_reached'] is True:
    #     plt.xlabel('cell number')
    #     plt.ylabel('cell size [m]')

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # plt.plot(state['mean_free_path'] / state['mirror_cell_sizes'], linestyle=linestyle, linewidth=linewidth, color='b')
    # if state['termination_criterion_reached'] is True:
    #     plt.xlabel('cell number')
    #     plt.ylabel('mfp / cell size')

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # plt.plot(z_array, state['v_th'], linestyle=linestyle, linewidth=linewidth, color='b',
    #          label='$v_{th}$' + label_suffix)
    # plt.plot(z_array, state['U'], linestyle=linestyle, linewidth=linewidth, color='r',
    #          label='$U_{MMM}$' + label_suffix)
    # plt.plot(z_array, state['v_col'], linestyle=linestyle, linewidth=linewidth, color='c',
    #          label='$v_{col}$' + label_suffix)
    # if state['termination_criterion_reached'] is True:
    #     plt.ylabel('velocity [m/s]')
    #     plt.xlabel(xlabel)

    fig_cnt += 1
    plt.figure(fig_cnt)
    plt.plot(state['t_evolution'], state['flux_normalized_std_evolution'], linestyle=linestyle, linewidth=linewidth,
             label='normalized flux std' + label_suffix, color='k')
    if state['termination_criterion_reached'] is True:
        plt.ylabel('flux normalized std evolution')
        plt.xlabel('simulation time')

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # plt.plot(state['t_evolution'], state['flux_min_evolution'], linestyle=linestyle, linewidth=linewidth,
    #          label='flux max' + label_suffix, color='r')
    # plt.plot(state['t_evolution'], state['flux_max_evolution'], linestyle=linestyle, linewidth=linewidth,
    #          label='flux min' + label_suffix, color='b')
    # if state['termination_criterion_reached'] is True:
    #     plt.ylabel('fluxes min/max evolution')
    #     plt.xlabel('simulation time')

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # plt.plot(z_array, state['n_c'] / state['n_tR'], linewidth=linewidth, linestyle=linestyle, color='r',
    #          label='n_c/n_tR' + label_suffix)
    # plt.plot(z_array, state['n_tL'] / state['n_tR'], linewidth=linewidth, linestyle=linestyle, color='g',
    #          label='n_tL/n_tR' + label_suffix)
    # if state['termination_criterion_reached'] is True:
    #     plt.ylabel('density ratios')
    #     plt.xlabel(xlabel)

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # inds = np.where(state['alpha_tR'] > 0)
    # plt.plot(z_array[inds], state['n_tR'][inds] / state['n'][inds] / state['alpha_tR'][inds], linestyle=linestyle,
    #          label='$n/\\alpha$ tR fraction' + label_suffix,
    #          linewidth=linewidth, color='b')
    # inds = np.where(state['alpha_tL'] > 0)
    # plt.plot(z_array[inds], state['n_tL'][inds] / state['n'][inds] / state['alpha_tL'][inds], linestyle=linestyle,
    #          label='$n/\\alpha$ tL fraction' + label_suffix,
    #          linewidth=linewidth, color='g')
    # inds = np.where(state['alpha_c'] > 0)
    # plt.plot(z_array[inds], state['n_c'][inds] / state['n'][inds] / state['alpha_c'][inds], linestyle=linestyle,
    #          label='$n/\\alpha$ c fraction' + label_suffix,
    #          linewidth=linewidth, color='r')
    # if state['termination_criterion_reached'] is True:
    #     plt.ylabel('$n/n_{tot}/\\alpha$ ratios')
    #     plt.xlabel(xlabel)
    #     plt.ylim([0, 2])

    # fig_cnt += 1
    # plt.figure(fig_cnt)
    # plt.plot(z_array, state['flux_E'], linestyle=linestyle, label='flux_E' + label_suffix,
    #          linewidth=linewidth, color='b')
    # if state['termination_criterion_reached'] is True:
    #     plt.ylabel('energy flux')
    #     plt.xlabel(xlabel)

    settings['num_plots'] = fig_cnt

    if state['termination_criterion_reached'] is True:
        for fig_num in range(1, fig_cnt + 1):
            plt.figure(fig_num)
            plt.tight_layout()
            plt.grid(True)

    return settings

def save_plots(settings):
    for fig_num in range(1, settings['num_plots'] + 1):
        plt.figure(fig_num)
        # create save directory
        if not os.path.exists(settings['save_dir']):
            logging.info('Creating save directory: ' + settings['save_dir'])
        plt.savefig(settings['save_dir'] + '/fig' + str(fig_num))
    return


def num_to_coords_str(x, notation_type='both', scilimits=(-1e5, 1e6)):
    if notation_type in ['f', 'float']:
        format_type = 'f'
    elif notation_type in ['e', 'scientific']:
        format_type = 'e'
    else:
        if x < scilimits[0] or x > scilimits[1]:
            format_type = 'e'
        else:
            format_type = 'f'
    return ('{:.4' + format_type + '}').format(x)


def format_coord(x, y, X, Y, Z, **notation_kwargs):
    coords_str = 'x=' + num_to_coords_str(x, **notation_kwargs) + ', y=' + num_to_coords_str(y, **notation_kwargs)
    if len(X.shape) == 1 and len(Y.shape) == 1:
        xarr, yarr = X, Y
    elif len(X.shape) == 2 and len(Y.shape) == 2:
        xarr = X[0, :]
        yarr = Y[:, 0]
    else:
        raise ValueError('invalid X, Y formats.')

    if ((x > xarr.min()) & (x <= xarr.max()) &
            (y > yarr.min()) & (y <= yarr.max())):
        col = np.searchsorted(xarr, x) - 1
        row = np.searchsorted(yarr, y) - 1
        z = Z[row, col]
        coords_str += ', z=' + num_to_coords_str(z, **notation_kwargs) + '  [' + str(row) + ',' + str(col) + ']'
    return coords_str


def update_format_coord(X, Y, Z, ax=None, **notation_kwargs):
    if ax == None:
        ax = plt.gca()
    ax.format_coord = lambda x, y: format_coord(x, y, X, Y, Z, **notation_kwargs)
    plt.show()
