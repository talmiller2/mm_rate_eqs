import matplotlib

# matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.size': 12})

import numpy as np
from scipy.optimize import curve_fit

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation
from mm_rate_eqs.fusion_functions import get_lawson_parameters

from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss, get_magnetic_pressure, \
    get_ideal_gas_pressure, get_ideal_gas_energy_per_volume, get_magnetic_field_for_given_pressure, \
    get_bohm_diffusion_constant, get_larmor_radius

plt.close('all')

main_dir = '/Users/talmiller/Downloads/mm_rate_eqs/'

main_dir += '/runs/slurm_runs/set38_RFscat_ni_1e20_T_10keV_N_20/'
# main_dir += '/runs/slurm_runs/set39_RFscat_ni_1e20_T_10keV_N_10/'


nu_RF_c_list = [0.05, 0.1, 0.5, 0.05, 0.1, 0.5, 0.05, 0.1, 0.5]
nu_RF_tL_list = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3]
nu_RF_tR_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# nu_RF_c_list  = [0.1, 0.1, 0.1]
# nu_RF_tL_list = [1.0, 0.5, 0.3]
# nu_RF_tR_list = [1.0, 1.0, 1.0]


if len(nu_RF_c_list) != len(nu_RF_tL_list) or len(nu_RF_c_list) != len(nu_RF_tR_list) \
        or len(nu_RF_tL_list) != len(nu_RF_tR_list):
    raise ValueError('lengths incompatible.')

nu_RF_factor_list = [0.1, 0.2, 0.5, 0.8, 1, 2, 5, 10, 15, 20, 25, 30]

color_list = ['b', 'b', 'b', 'g', 'g', 'g', 'r', 'r', 'r']
linestyle_list = ['-', '--', ':', '-', '--', ':', '-', '--', ':']

linewidth = 3

for ind_set, (nu_RF_c, nu_RF_tL, nu_RF_tR) in enumerate(zip(nu_RF_c_list, nu_RF_tL_list, nu_RF_tR_list)):
    flux_list = np.nan * np.zeros(len(nu_RF_factor_list))

    label = ''
    label += 'nu_RF_c_' + str(nu_RF_c)
    label += '_tL_' + str(nu_RF_tL)
    label += '_tR_' + str(nu_RF_tR)

    for ind_fac, nu_RF_factor in enumerate(nu_RF_factor_list):

        run_name = label
        run_name += '_fac_' + str(nu_RF_factor)

        save_dir = main_dir + '/' + run_name

        state_file = save_dir + '/state.pickle'
        settings_file = save_dir + '/settings.pickle'
        try:
            state, settings = load_simulation(state_file, settings_file)

            # post process the flux normalization
            ni = state['n'][0]
            Ti_keV = state['Ti'][0] / 1e3
            _, flux_lawson = get_lawson_parameters(ni, Ti_keV, settings)
            state['flux_mean'] *= settings['cross_section_main_cell']
            flux_list[ind_fac] = state['flux_mean'] / flux_lawson

        except:
            pass

    plt.figure(1)
    label_plot = 'c=' + str(nu_RF_c) + ', tL=' + str(nu_RF_tL)
    plt.plot(nu_RF_factor_list, flux_list, label=label_plot,
             linestyle=linestyle_list[ind_set], color=color_list[ind_set], linewidth=2)

fig = plt.figure(1)
plt.yscale("log")
# plt.xscale("log")
plt.xlabel('$\\nu_{RF}$ factor')
plt.ylabel('$\\phi_{axial}/\\phi_{Lawson}$')
plt.tight_layout()
plt.grid(True)
plt.legend()
