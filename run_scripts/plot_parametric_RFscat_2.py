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

main_dir += '/runs/slurm_runs/set40_RFscat_ni_1e20_T_10keV_N_20/'

nu_RF_tL_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
nu_RF_c = 0.1
nu_RF_tR = 1.0
nu_RF_factor_list = [1, 10]

selectivity = 1 / np.array(nu_RF_tL_list)

color_list = ['b', 'g']
linestyle_list = ['-', '-']
linewidth = 3

for ind_fac, nu_RF_factor in enumerate(nu_RF_factor_list):

    flux_list = np.nan * np.zeros(len(nu_RF_tL_list))

    for ind_tR, nu_RF_tL in enumerate(nu_RF_tL_list):

        run_name = ''
        run_name += 'nu_RF_c_' + str(nu_RF_c)
        run_name += '_tL_' + str(nu_RF_tL)
        run_name += '_tR_' + str(nu_RF_tR)
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
            flux_list[ind_tR] = state['flux_mean'] / flux_lawson

        except:
            pass

    plt.figure(1)
    # label_plot = 'c=' + str(nu_RF_c) + ', $\\nu_{RF,fac}$=' + str(nu_RF_factor)
    label_plot = '$\\nu_{RF,fac}$=' + str(nu_RF_factor)
    # plt.plot(nu_RF_tL_list, flux_list, label=label_plot,
    #          linestyle=linestyle_list[ind_fac], color=color_list[ind_fac], linewidth=2)
    plt.plot(selectivity, flux_list, label=label_plot,
             linestyle=linestyle_list[ind_fac], color=color_list[ind_fac], linewidth=2)

fig = plt.figure(1)
plt.yscale("log")
# plt.xscale("log")
# plt.xlabel('$\\nu_{RF,tL}$')
plt.xlabel('selectivity = $1/\\nu_{RF,tL}$')
plt.ylabel('$\\phi_{axial}/\\phi_{Lawson}$')
plt.title('$\\nu_{RF,tR}=1,\\nu_{RF,c}=0.1,N=20$')
plt.tight_layout()
plt.grid(True)
plt.legend()
