import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 16})

from default_settings import define_default_settings
from relaxation_algorithm_functions import load_simulation

# parametric scan
# save_dir_main = 'runs/runs_April_2020/runs_no_transition_different_number_of_cells/'
# number_of_cells_list = np.round(np.linspace(5,150,15))[0:-1]

# nullify_ntL_factor = 0.05
# # nullify_ntL_factor = 0.01
# label = 'isothermal_nullify_ntL_factor_' + str(nullify_ntL_factor)
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_nullify_ntL_factor_' \
#                 + str(nullify_ntL_factor)
# label = 'isothermal_rbc_none_energycons_none_U_0'
# label = 'isothermal U=0'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none'
# label = 'isothermal_rbc_none_energycons_none_U_0.3'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_U_0.3'


plt.close('all')

mode = 'iso2'

colors = []
mfps = []
colors += ['b']
mfps += [1]
colors += ['g']
mfps += [5]
# colors += ['r']
# mfps += [20]

for color, mfp in zip(colors, mfps):
    print('mode = ' + mode)

    # d = 3
    d = 1

    U = 0
    linestyle = '-'
    # U = 0.3
    # linestyle = '--'

    label = ''
    save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none'
    if mode == 'iso':
        label += 'isothermal'
        save_dir_main += '_iso'
    elif mode == 'iso2':
        label += 'isothermal iso-mfp'
        save_dir_main += '_iso2'
    elif mode == 'cool':
        label += 'cool d=' + str(d)
        save_dir_main += '_cool_d_' + str(d)
    label += ', U=' + str(U)
    save_dir_main += '_U_' + str(U)

    if mfp > 1:
        save_dir_main += '_mfpX' + str(mfp)

    number_of_cells_list = np.round(np.linspace(5, 100, 15))

    flux_list = np.nan * np.zeros(len(number_of_cells_list))

    for i, number_of_cells in enumerate(number_of_cells_list):
        # print('number_of_cells=' + str(int(number_of_cells)))

        save_dir = save_dir_main + '/number_of_cells_' + str(int(number_of_cells))

        try:  # will fail if run does not exist
            state_file = save_dir + '/state.pickle'
            settings_file = save_dir + '/settings.pickle'
            state, settings = load_simulation(state_file, settings_file)
            flux_list[i] = state['flux_mean']

        except:
            pass

        # extract the density profile
        chosen_num_cells = 32
        if number_of_cells == chosen_num_cells:
            plt.figure(2)
            plt.plot(state['n'], '-', label=label, linestyle=linestyle, color=color)

    plt.figure(1)
    plt.plot(number_of_cells_list, flux_list, '-', label=label, linestyle=linestyle, color=color)
    # plt.semilogx(number_of_cells_list, flux_list, '-', label=label, linestyle=linestyle, color=color)
    # plt.semilogy(number_of_cells_list, flux_list, '-', label=label, linestyle=linestyle, color=color)
    # plt.loglog(number_of_cells_list, flux_list, '-', label=label, linestyle=linestyle, color=color)

    # p = np.polyfit(number_of_cells_list, 1/flux_list, deg=1)
    # flux_fit = 1/np.polyval(p, number_of_cells_list)
    # plt.plot(number_of_cells_list, flux_fit, '-', label='1/N fit')

    # clear nans for fit
    norm_factor = 1e27
    inds_flux_not_nan = [i for i in range(len(flux_list)) if not np.isnan(flux_list[i])]
    n_cells = number_of_cells_list[inds_flux_not_nan]
    flux_cells = flux_list[inds_flux_not_nan] / norm_factor

    # fit_function = lambda x, a, b: a + b / x
    # fit_function = lambda x, a, b: a + b / x ** 2
    fit_function = lambda x, a, b, gamma: a + b / x ** gamma
    # fit_function = lambda x, b, gamma: b / x ** gamma
    # fit_function = lambda x, a: a / x
    # fit_function = lambda x, a: a / x ** 2.0
    # fit_function = lambda x, a, b: a * x + b
    popt, pcov = curve_fit(fit_function, n_cells, flux_cells)
    flux_cells_fit = fit_function(n_cells, *popt) * norm_factor
    # plt.plot(n_cells, flux_cells_fit, label=label + ' fit', linestyle='--', color=color)
    plt.plot(n_cells, flux_cells_fit, label=label + ', fit power = ' + '{:0.2f}'.format(popt[-1]), linestyle='--',
             color=color)

plt.figure(1)
plt.xlabel('number of cells')
plt.ylabel('flux')
plt.title('flux as a function of system size')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(2)
plt.xlabel('cell number')
plt.ylabel('density')
plt.title('density profile for N = ' + str(chosen_num_cells) + ' cells')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.grid(True)
