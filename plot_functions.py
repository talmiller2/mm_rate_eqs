import matplotlib.pyplot as plt
import numpy as np


def plot_relaxation_status(state, settings):
    if state['termination_criterion_reached'] is True:
        linewidth = 3
    else:
        linewidth = 1
    label_suffix = ''

    z_array = np.cumsum(state['mirror_cell_sizes'])

    plt.figure(1)
    plt.plot(z_array, state['n_tL'], linewidth=linewidth, linestyle='-', color='g', label='n_tL' + label_suffix)
    plt.plot(z_array, state['n_tR'], linewidth=linewidth, linestyle='-', color='b', label='n_tR' + label_suffix)
    plt.plot(z_array, state['n_c'], linewidth=linewidth, linestyle='-', color='r', label='n_c' + label_suffix)
    plt.plot(z_array, state['n'], linewidth=linewidth, linestyle='-', color='k', label='n' + label_suffix)

    plt.figure(2)
    plt.plot(z_array, state['flux_trans_R'], linestyle='-', label='flux_trans_R' + label_suffix, linewidth=linewidth,
             color='b')
    plt.plot(z_array, state['flux_trans_L'], linestyle='-', label='flux_trans_L' + label_suffix, linewidth=linewidth,
             color='r')
    plt.plot(z_array, state['flux_mmm_drag'], linestyle='-', label='flux_mmm_drag' + label_suffix, linewidth=linewidth,
             color='g')
    plt.plot(z_array, state['flux'], linestyle='-', label='flux total' + label_suffix, linewidth=linewidth, color='k')

    plt.figure(3)
    plt.plot(z_array, state['coulomb_scattering_rate'], linestyle='-', label='scattering rate' + label_suffix,
             linewidth=linewidth, color='b')
    plt.plot(z_array, state['transmission_rate'], linestyle='-', label='transmission rate' + label_suffix,
             linewidth=linewidth, color='g')
    plt.plot(z_array, state['mmm_drag_rate'], linestyle='-', label='MMM drag rate' + label_suffix, linewidth=linewidth,
             color='r')

    plt.figure(4)
    plt.plot(z_array, state['Ti'] / settings['keV'], linewidth=linewidth, linestyle='-', color='r', label='Ti' + label_suffix)
    plt.plot(z_array, state['Te'] / settings['keV'], linewidth=linewidth, linestyle='-', color='b', label='Te' + label_suffix)

    plt.figure(5)
    plt.plot(z_array, state['mean_free_path'], linestyle='-', label='mfp' + label_suffix, linewidth=linewidth, color='r')

    plt.figure(6)
    plt.plot(z_array, state['alpha_tR'], linestyle='-', label='$\\alpha_{tR}$' + label_suffix, linewidth=linewidth, color='b')
    plt.plot(z_array, state['alpha_tL'], linestyle='-', label='$\\alpha_{tL}$' + label_suffix, linewidth=linewidth, color='g')
    plt.plot(z_array, state['alpha_c'], linestyle='-', label='$\\alpha_{c}$' + label_suffix, linewidth=linewidth, color='r')

    plt.figure(7)
    plt.plot(z_array, linestyle='-', linewidth=linewidth, color='b')

    return


def plot_relaxation_end(title = '', show_legend = False):
    plt.figure(1)
    plt.ylabel('$m^{-3}$')
    plt.xlabel('z [m]')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    if show_legend is True: plt.legend()

    plt.figure(2)
    plt.ylabel('flux')
    plt.xlabel('z [m]')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    if show_legend is True: plt.legend()

    plt.figure(3)
    plt.ylabel('rate [$s^{-1}$]')
    plt.xlabel('z [m]')
    plt.yscale('log')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    if show_legend is True: plt.legend()

    plt.figure(4)
    plt.ylabel('T [keV]')
    plt.xlabel('z [m]')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    if show_legend is True: plt.legend()

    plt.figure(5)
    plt.ylabel('mfp [m]')
    plt.xlabel('z [m]')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    if show_legend is True: plt.legend()

    plt.figure(6)
    plt.ylabel('$\\alpha$')
    plt.xlabel('z [m]')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    if show_legend is True: plt.legend()

    plt.figure(7)
    plt.xlabel('cell number')
    plt.ylabel('cell length [m]')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    if show_legend is True: plt.legend()


    # plt.figure(3)
    # plt.plot(flux_i_var_list, label='i flux', color='r')
    # #        plt.ylim([0, 2])
    # plt.legend()
    # plt.ylabel('fluxes variance')
    # plt.xlabel('print step')
    # plt.title(label_U)
    # plt.tight_layout()
    # plt.grid()
    #
    # plt.figure(6 + num_plots * ind_U + 2 * num_plots * ind_Rm)
    # plt.plot(flux_i_max_list, '-r', label='i flux max')
    # plt.plot(flux_i_min_list, '--r', label='i flux min')
    # #        plt.yscale('log')
    # #        plt.ylim([-1e24, 5e27])
    # #    df = max(flux_i_max_list[-1],flux_e_max_list[-1])
    # #    plt.ylim([min(flux_i_min_list[-1],flux_e_min_list[-1])-df, max(flux_i_max_list[-1],flux_e_max_list[-1])+df])
    # plt.legend()
    # plt.ylabel('fluxes')
    # plt.xlabel('print step')
    # plt.title(label_U)
    # plt.tight_layout()
    # plt.grid()

    return