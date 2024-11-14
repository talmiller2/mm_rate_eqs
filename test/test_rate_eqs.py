import matplotlib

matplotlib.use('TkAgg')  # to avoid macOS bug where plots cant get minimized

import matplotlib.pyplot as plt
import os

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.relaxation_algorithm_functions import find_rate_equations_steady_state
from mm_rate_eqs.fusion_functions import get_lawson_parameters

plt.close('all')

### test the algorithm
settings = {}
settings['fontsize'] = 12

settings['gas_name'] = 'hydrogen'
# settings['gas_name'] = 'DT-mix'

# settings['save_state'] = 'False'
settings['assume_constant_density'] = False
# settings['assume_constant_density'] = True
# settings['assume_constant_temperature'] = False
settings['assume_constant_temperature'] = True
# settings['ion_scattering_rate_factor'] = 10

settings['cell_size'] = 1
# settings['cell_size'] = 3
# settings['cell_size'] = 10

settings['plasma_dimension'] = 1
# settings['plasma_dimension'] = 1.5
# settings['plasma_dimension'] = 2
# settings['plasma_dimension'] = 3
# settings['plasma_dimension'] = 10
# settings['plasma_dimension'] = 100
# settings['number_of_cells'] = 10
settings['number_of_cells'] = 20  # nominal value
# settings['number_of_cells'] = 30
# settings['number_of_cells'] = 40
# settings['number_of_cells'] = 50
# settings['number_of_cells'] = 100
# settings['number_of_cells'] = 150
# settings['number_of_cells'] = 200

# settings['n0'] = 2e22  # m^-3
settings['n0'] = 1e21  # m^-3
# settings['n0'] = 1e20  # m^-3

# settings['Ti_0'] = 3 * 1e3 # eV
# settings['Te_0'] = 3 * 1e3 # eV
settings['Ti_0'] = 10 * 1e3  # eV
settings['Te_0'] = 10 * 1e3  # eV

# settings['right_scat_factor'] = 1.0
# settings['right_scat_factor'] = 10.0
# settings['right_scat_factor'] = 100.0

settings['U0'] = 0
# settings['U0'] = 0.01
# settings['U0'] = 0.02
# settings['U0'] = 0.05
# settings['U0'] = 0.1
# settings['U0'] = 0.2
# settings['U0'] = 0.3
# settings['U0'] = 0.5
# settings['U0'] = 0.8

# settings['flux_normalized_termination_cutoff'] = 0.5
# settings['flux_normalized_termination_cutoff'] = 0.1
# settings['flux_normalized_termination_cutoff'] = 0.05
# settings['flux_normalized_termination_cutoff'] = 0.03
settings['flux_normalized_termination_cutoff'] = 0.01
# settings['flux_normalized_termination_cutoff'] = 1e-4

# settings['alpha_definition'] = 'geometric_constant'
# settings['alpha_definition'] = 'geometric_local'

# settings['U_for_loss_cone_factor'] = 1.0
# settings['U_for_loss_cone_factor'] = 0.5

# settings['adaptive_mirror'] = 'adjust_cell_size_with_mfp'
# settings['adaptive_mirror'] = 'adjust_cell_size_with_vth'

settings['right_boundary_condition'] = 'none'
# settings['right_boundary_condition'] = 'adjust_ntL_for_nend'
# settings['right_boundary_condition'] = 'adjust_ntR_for_nend'
# settings['right_boundary_condition'] = 'adjust_nc_for_nend'
# settings['right_boundary_condition'] = 'adjust_all_species_for_nend'
# settings['right_boundary_condition'] = 'nullify_ntL'

# settings['right_boundary_condition_density_type'] = 'none'
# settings['right_boundary_condition_density_type'] = 'n_transition'
# settings['right_boundary_condition_density_type'] = 'n_expander'

# settings['n_expander_factor'] = 0.5
# settings['n_expander_factor'] = 0.1
# settings['n_expander_factor'] = 0.05
# settings['n_expander_factor'] = 0.01

# settings['time_step_definition_using_species'] = 'all'
# settings['time_step_definition_using_species'] = 'only_c_tR'

# settings['nullify_ntL_factor'] = 0.05
# settings['nullify_ntL_factor'] = 0.05
# settings['nullify_ntL_factor'] = 0.01

settings['transition_type'] = 'none'
# settings['transition_type'] = 'smooth_transition_to_free_flow'

# settings['energy_conservation_scheme'] = 'none'
# settings['energy_conservation_scheme'] = 'simple'
# settings['energy_conservation_scheme'] = 'detailed'

# settings['dt_status'] = 1e-5
# settings['dt_status'] = 1e-4
# settings['dt_status'] = 1e-3
settings['time_steps_status'] = int(1e3)

settings['use_RF_terms'] = True

# settings['RF_cl'] = 0
# settings['RF_cr'] = 0
# settings['RF_lc'] = 0
# settings['RF_rc'] = 0

# settings['RF_cl'] = 0.03 # set32_B0_1T_l_1m_Post_Rm_3_intervals, E_RF_kVm = 25 kV/m, alpha = 1.0, beta = 0.0
# settings['RF_cr'] = 0.03
# settings['RF_lc'] = 0.39
# settings['RF_rc'] = 0.39 # flux_axial_over_flux_lawson = 304.8318045226798

# settings['RF_cl'] = 0.03 # set32_B0_1T_l_1m_Post_Rm_3_intervals, E_RF_kVm = 25 kV/m, alpha = 0.9, beta = -1.0
# settings['RF_cr'] = 0.01
# settings['RF_lc'] = 0.1
# settings['RF_rc'] = 0.35 # flux_axial_over_flux_lawson = 84.7490096952208

# settings['RF_cl'] = 0.025 # set32_B0_1T_l_1m_Post_Rm_3_intervals, E_RF_kVm = 25 kV/m, alpha = 0.8, beta = -5.0
# settings['RF_cr'] = 0.004
# settings['RF_lc'] = 0.038
# settings['RF_rc'] = 0.324 # flux_axial_over_flux_lawson = 60.071421918239636

# settings['RF_cl'] = 0.023 # set32_B0_1T_l_1m_Post_Rm_3_intervals, E_RF_kVm = 50 kV/m, alpha = 0.8, beta = -5.0
# settings['RF_cr'] = 0.008
# settings['RF_lc'] = 0.074
# settings['RF_rc'] = 0.484 # flux_axial_over_flux_lawson = 24.98291050008519

# settings['RF_cl'] = 0.013 # set32_B0_1T_l_1m_Post_Rm_3_intervals, E_RF_kVm = 100 kV/m, alpha = 0.8, beta = -5.0
# settings['RF_cr'] = 0.016
# settings['RF_lc'] = 0.132
# settings['RF_rc'] = 0.638 # flux_axial_over_flux_lawson = 43.693868033664494

# settings['RF_cl'] = 0.027 # set33_B0_1T_l_3m_Post_Rm_3_intervals, E_RF_kVm = 100 kV/m, alpha = 0.8, beta = -5.0
# settings['RF_cr'] = 0.023
# settings['RF_lc'] = 0.215
# settings['RF_rc'] = 0.953 # flux_axial_over_flux_lawson = 10.186296048238205 (N=50: 0.0442)

# settings['RF_cl'] = 0.018 # set32_B0_1T_l_1m_Post_Rm_3_intervals, E_RF_kVm = 100 kV/m, alpha = 0.8, beta = -10.0
# settings['RF_cr'] = 0.011
# settings['RF_lc'] = 0.094
# settings['RF_rc'] = 0.392 # flux_axial_over_flux_lawson = 221.05007062678922

# settings['RF_cl'] = 0.078 # set32_B0_1T_l_1m_Post_Rm_3_intervals, B_RF = 0.04 T, alpha = 0.8, beta = -5.0
# settings['RF_cr'] = 0.015
# settings['RF_lc'] = 0.074
# settings['RF_rc'] = 0.55 # flux_axial_over_flux_lawson = 23.49605415012276 (with N=20)

# settings['RF_cl'] = 0.02 # artificial test
# settings['RF_cr'] = 0.02
# settings['RF_lc'] = 0.2
# settings['RF_rc'] = 0.2 # flux_axial_over_flux_lawson = 26.97957449292672

# settings['RF_cl'] = 0.2 # artificial test
# settings['RF_cr'] = 0.2
# settings['RF_lc'] = 0.2
# settings['RF_rc'] = 0.2 # flux_axial_over_flux_lawson = 26.97957449292672

settings['RF_cl'] = 0.1  # artificial test
settings['RF_cr'] = 0.1
settings['RF_lc'] = 0.1
settings['RF_rc'] = 1.0  # flux_axial_over_flux_lawson = 26.97957449292672

# settings['RF_cl'] = 0.1 # artificial test
# settings['RF_cr'] = 0.05
# settings['RF_lc'] = 0.1
# settings['RF_rc'] = 1.0 # flux_axial_over_flux_lawson = 5.663855952708682

# settings['RF_cl'] = 0.02 # artificial test
# settings['RF_cr'] = 0.02
# settings['RF_lc'] = 0.1
# settings['RF_rc'] = 1.0 # flux_axial_over_flux_lawson = 7.058074719004123

# settings['RF_cl'] = 0.1 # artificial test
# settings['RF_cr'] = 0.02
# settings['RF_lc'] = 0.2
# settings['RF_rc'] = 1.0 # flux_axial_over_flux_lawson = 1.3713367871050115

# settings['RF_cl'] = 0.025 # artificial test
# settings['RF_cr'] = 0.004
# settings['RF_lc'] = 0.2
# settings['RF_rc'] = 1.0 # flux_axial_over_flux_lawson = 0.31197067374025955

# settings['RF_cl'] = 0.025 # artificial test
# settings['RF_cr'] = 0.004
# settings['RF_lc'] = 0.038
# settings['RF_rc'] = 1.0 # flux_axial_over_flux_lawson = 0.3105717517410081

# settings['RF_cl'] = 0.025 # artificial test
# settings['RF_cr'] = 0.004
# settings['RF_lc'] = 0.038
# settings['RF_rc'] = 0.324

# settings['RF_cl'] = 0.025 # artificial test
# settings['RF_cr'] = 0.024
# settings['RF_lc'] = 0.324
# settings['RF_rc'] = 0.324

# settings['RF_cl'] += 0.01 # artificial addition of sometime that has no selectivity but enhances the rates
# settings['RF_cr'] += 0.01
# settings['RF_lc'] += 0.1
# settings['RF_rc'] += 0.1

# fac = 0
# fac = 0.2
fac = 1
# fac = 10
# fac = 50
# fac = 100
# fac = 200

settings['RF_cl'] *= fac
settings['RF_cr'] *= fac
settings['RF_lc'] *= fac
settings['RF_rc'] *= fac

settings = define_default_settings(settings)
# settings['n_end_min'] = 0.3 * settings['n0']

# settings['save_dir'] = '../runs/runs_August_2020/'
# settings['save_dir'] = '../runs/runs_September_2020/'
# settings['save_dir'] = '../runs/runs_October_2020/'
settings['save_dir'] = '../runs/tests/'

os.makedirs(settings['save_dir'], exist_ok=True)

settings['save_dir'] += 'test'

settings['save_dir'] += '_N_' + str(settings['number_of_cells'])
settings['save_dir'] += '_U_' + str(settings['U0'])

if settings['adaptive_mirror'] != 'none':
    settings['save_dir'] += '_adap_' + settings['adaptive_mirror']

if settings['transition_type'] == 'none':
    settings['save_dir'] += '_trans_none'
else:
    settings['save_dir'] += '_trans_smooth'

if settings['assume_constant_temperature'] == True:
    settings['save_dir'] += '_isoT'
else:
    settings['save_dir'] += '_cool_d_' + str(settings['plasma_dimension'])

if settings['assume_constant_density'] == True:
    settings['save_dir'] += '_const_dens'

if settings['right_boundary_condition'] == 'nullify_ntL':
    settings['save_dir'] += '_rbc_nullify_ntL'
    settings['save_dir'] += '_factor_' + str(settings['nullify_ntL_factor'])
# elif settings['right_boundary_condition'] in ['adjust_ntL_for_nend', 'adjust_ntR_for_nend',
#                                               'adjust_all_species_for_nend']:
elif 'adjust' in settings['right_boundary_condition']:
    settings['save_dir'] += '_' + settings['right_boundary_condition']
    settings['save_dir'] += '_nend_' + str(settings['n_expander_factor'])
elif settings['right_boundary_condition'] == 'none':
    settings['save_dir'] += '_rbc_none'

# settings['save_dir'] += '_energy_scheme_' + settings['energy_conservation_scheme']

if settings['alpha_definition'] == 'geometric_constant':
    settings['save_dir'] += '_constLC'

if settings['U_for_loss_cone_factor'] != 1.0:
    settings['save_dir'] += '_Ufac' + str(settings['U_for_loss_cone_factor'])

if settings['time_step_definition_using_species'] == 'only_c_tR':
    settings['save_dir'] += '_timestep_def_without_tL'

# settings['save_dir'] += '_nmin0'

# settings['dt_factor'] = 0.1 / 3.0
# settings['save_dir'] += '_dt_factor_3'

# settings['max_num_time_steps'] = 1000
# settings['save_dir'] = '../runs/runs_August_2020/TEST'

# settings['n_min'] = 1e5
# settings['n_min'] = 1e19
# settings['n_min'] = 1e18
# settings['n_min'] = settings['n0'] * 1e-4
# settings['n_min'] = settings['n0'] * 1e-10
# settings['save_dir'] += '_nmin_' + str('{:.2e}'.format(settings['n_min']))

# settings['mfp_min'] = 1e-2
# settings['save_dir'] += '_mfp_min_' + str(settings['mfp_min'])

# settings['t_stop'] = 10e-4
# settings['max_num_time_steps'] = int(2e4) - 1

print('save dir: ' + str(settings['save_dir']))

# settings['save_state'] = True
settings['save_state'] = False

state = find_rate_equations_steady_state(settings)

ni = state['n'][0]
Ti_keV = state['Ti'][0] / 1e3
_, flux_lawson = get_lawson_parameters(ni, Ti_keV, settings)
flux_axial_over_flux_lawson = state['flux_mean'] * settings['cross_section_main_cell'] / flux_lawson
print('flux_axial_over_flux_lawson = ' + str(flux_axial_over_flux_lawson))

plt.figure(1)
title = ''
# title += '$\\bar{N}_{LC \\rightarrow T}=$' + str(settings['RF_rc'])
# title += ', $\\bar{N}_{T \\rightarrow LC}=$' + str(settings['RF_cr']) + ', '
title += '$\\bar{N}_{rc}=$' + str(settings['RF_rc'])
title += ', $\\bar{N}_{lc}=$' + str(settings['RF_lc'])
title += ', $\\bar{N}_{cr}=$' + str(settings['RF_cr'])
title += ', $\\bar{N}_{cl}=$' + str(settings['RF_cl']) + ', '
title += '$\\phi_{ss}/\\phi_{Lawson}=$' + '{:.2f}'.format(flux_axial_over_flux_lawson)
plt.title(title, fontsize=12)
