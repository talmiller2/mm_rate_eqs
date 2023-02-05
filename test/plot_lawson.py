import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 12
plt.close('all')

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.fusion_functions import get_lawson_parameters, get_fusion_power, get_fusion_charged_power, \
    get_sigma_v_fusion, get_reaction_label, get_lawson_criterion_piel
from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss, get_magnetic_pressure, \
    get_ideal_gas_pressure, get_ideal_gas_energy_per_volume, get_magnetic_field_for_given_pressure, \
    get_bohm_diffusion_constant, get_larmor_radius, get_alfven_wave_group_velocity, get_larmor_frequency
from mm_rate_eqs.rate_functions import calculate_coulomb_logarithm, get_thermal_velocity, get_coulomb_scattering_rate

from mm_rate_eqs.constants_functions import define_electron_mass, define_proton_mass, define_factor_eV_to_K, \
    define_boltzmann_constant, define_factor_Pa_to_bar

# fusion plasma
settings = {'gas_name': 'DT_mix'}
# settings = {'gas_name': 'hydrogen'}
# settings = {'gas_name': 'tritium'}
T = 10000.0
n = 2e21

settings['length_main_cell'] = 100  # m
settings['diameter_main_cell'] = 1  # m

settings = define_default_settings(settings=settings)
# settings['volume_main_cell'] = 1.0

Ti = T
Te = T
Ti_keV = Ti / 1e3
Te_keV = Te / 1e3
ne = n / 2
ni = n / 2

# Ti_keV_list = np.logspace(0.1, 2.5, 1000)
Ti_keV_list = np.linspace(1, 300, 1000)
reaction_list = ['D_T_to_n_alpha', 'D_D_to_p_T_n_He3', 'D_He3_to_p_alpha', 'p_B_to_3alpha']
# reaction_list = ['D_T_to_n_alpha']
Ti_keV_list_for_reactivity = [13.6, 16.0, 59.1, 137.2]
color_list = ['b', 'g', 'r', 'k']
for ind_reaction, (reaction, color) in enumerate(zip(reaction_list, color_list)):
    tau_lawson_list = []
    for curr_Ti_keV in Ti_keV_list:
        tau_lawson, flux_lawson = get_lawson_parameters(ni, curr_Ti_keV, settings, reaction=reaction)
        tau_lawson_list += [tau_lawson]
    tau_lawson_list = np.array(tau_lawson_list)

    metric = ni * Ti_keV_list * tau_lawson_list
    # metric = 1 / (ni * Ti_keV_list * tau_lawson_list)
    ind_min = np.argmin(metric)
    # ind_min = np.argmax(metric)
    # label = reaction
    label = get_reaction_label(reaction=reaction)
    label += ', T=' + '{:.1f}'.format(Ti_keV_list[ind_min]) + 'keV'
    plt.figure(1)
    plt.plot(Ti_keV_list, metric, label=label, linewidth=2, color=color)
    plt.plot(Ti_keV_list[ind_min], metric[ind_min], 'k', marker='o')
    print(reaction, ', min(nTtau)=', metric[ind_min], ' at $T_{min}$=', '{:.1f}'.format(Ti_keV_list[ind_min]) + 'keV')

    # # add the Lawson cretirion plot from Piel (2007) book, page 105
    # tau_lawson_piel_list = []
    # tau_lawson_ignition_piel_list = []
    # for curr_Ti_keV in Ti_keV_list:
    #     tau_lawson_piel, tau_lawson_ignition_piel = get_lawson_criterion_piel(ni, curr_Ti_keV, settings, eta=0.154,
    #                                                                           reaction=reaction)
    #     tau_lawson_piel_list += [tau_lawson_piel]
    #     tau_lawson_ignition_piel_list += [tau_lawson_ignition_piel]
    # tau_lawson_piel_list = np.array(tau_lawson_piel_list)
    # tau_lawson_ignition_piel_list = np.array(tau_lawson_ignition_piel_list)
    # metric2 = ni * Ti_keV_list * tau_lawson_piel_list
    # metric3 = ni * Ti_keV_list * tau_lawson_ignition_piel_list
    # label = get_reaction_label(reaction=reaction)
    # plt.plot(Ti_keV_list, metric2, label=label, linewidth=2, color=color, linestyle='--')
    # plt.plot(Ti_keV_list, metric3, label=label, linewidth=2, color=color, linestyle=':')

    ##########

    metric = ni * tau_lawson_list
    ind_min = np.argmin(metric)
    # label = reaction
    label = get_reaction_label(reaction=reaction)
    label += ', $T_{min}$=' + '{:.1f}'.format(Ti_keV_list[ind_min]) + 'keV'
    plt.figure(2)
    plt.plot(Ti_keV_list, metric, label=label, linewidth=2, color=color)
    plt.plot(Ti_keV_list[ind_min], metric[ind_min], 'k', marker='o')

    #############

    sigma_v_fusion = get_sigma_v_fusion(Ti_keV_list, reaction=reaction)
    ind_marker = np.where(Ti_keV_list >= Ti_keV_list_for_reactivity[ind_reaction])[0][0]
    T_marker = Ti_keV_list[ind_marker]
    print(
        '<sigma*v> at T=' + str(Ti_keV_list_for_reactivity[ind_reaction]) + 'keV = ' + str(sigma_v_fusion[ind_marker]))
    plt.figure(3)
    # label = reaction
    label = get_reaction_label(reaction=reaction)
    plt.plot(Ti_keV_list, sigma_v_fusion, label=label, linewidth=2, color=color)
    plt.plot(Ti_keV_list[ind_marker], metric[ind_marker], 'k', marker='o')

plt.figure(1)
plt.xscale('log')
plt.yscale('log')
plt.title('Lawson $n T \\tau$')
plt.legend()
plt.grid(True)

plt.figure(2)
plt.xscale('log')
plt.yscale('log')
plt.title('Lawson $n \\tau$')
plt.legend()
plt.grid(True)

plt.figure(3)
plt.xscale('log')
plt.yscale('log')
plt.title('$\\sigma v$')
plt.legend()
plt.grid(True)
