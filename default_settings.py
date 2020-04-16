import numpy as np

from fusion_functions import define_plasma_parameters


def define_default_settings(settings=None):
    if settings == None:
        settings = {}

    #### physical constants
    settings['lnCoulombLambda'] = 10.0
    settings['eV'] = 1.0
    settings['keV'] = 1e3 * settings['eV']
    settings['eV_to_K'] = 1.16e4
    settings['MeV_to_J'] = 1e6 * 1.6e-19
    settings['kB_K'] = 1.380649e-23  # J/K
    settings['kB_eV'] = settings['kB_K'] * settings['eV_to_K']  # J/eV

    ### plasma parameters5
    if 'gas_name' not in settings:
        settings['gas_name'] = 'hydrogen'
    if 'ionization_level' not in settings:
        settings['ionization_level'] = 1.0
        # settings['ionization_level'] = None
    settings['me'], settings['mp'], settings['mi'], settings['A_atomic_weight'], settings['Z_ion'] \
        = define_plasma_parameters(gas_name=settings['gas_name'], ionization_level=settings['ionization_level'])

    ### system parameters
    if 'n0' not in settings:
        settings['n0'] = 1e22  # m^-3
    if 'Ti_0' not in settings:
        settings['Ti_0'] = 3 * settings['keV']
    if 'Te_0' not in settings:
        settings['Te_0'] = 1 * settings['keV']
    if 'B' not in settings:
        settings['B'] = 1.0 * np.sqrt(settings['n0'] / 1e20 * settings['Ti_0'] / (5 * settings['keV']))  # [Tesla]
    if 'Rm' not in settings:
        # settings['Rm'] = 1.4
        # settings['Rm'] = 2.0
        settings['Rm'] = 3.0
        # settings['Rm'] = 5.0
    if 'U0' not in settings:
        settings['U0'] = 0
    if 'mmm_velocity_type' not in settings:
        # settings['mmm_velocity_type'] = 'absolute'
        settings['mmm_velocity_type'] = 'relative_to_thermal_velocity'
    if 'transition_density_factor' not in settings:
        # settings['transition_density_factor'] = 0.5
        settings['transition_density_factor'] = 0.1
        # settings['transition_density_factor'] = 0.01
    if 'delta_n_smoothing_factor' not in settings:
        # settings['delta_n_smoothing_factor'] = 0.01
        settings['delta_n_smoothing_factor'] = 0.05
        # settings['delta_n_smoothing_factor'] = 0.1
        # settings['delta_n_smoothing_factor'] = 0.5
    if 'cell_size' not in settings:
        settings['cell_size'] = 3.0  # m (MMM wavelength)
    if 'number_of_cells' not in settings:
        # settings['number_of_cells'] = 30
        # settings['number_of_cells'] = 50
        # settings['number_of_cells'] = 100
        # settings['number_of_cells'] = 150
        settings['number_of_cells'] = 200
        # settings['number_of_cells'] = 300
        # settings['number_of_cells'] = 1000
    if 'length_main_cell' not in settings:
        settings['length_main_cell'] = 100  # m
    if 'diameter_main_cell' not in settings:
        settings['diameter_main_cell'] = 0.5  # m
    settings['cross_section_main_cell'] = np.pi * (settings['diameter_main_cell'] / 2) ** 2  # m^3
    settings['volume_main_cell'] = settings['length_main_cell'] * settings['cross_section_main_cell']  # m^3

    ### additional options
    if 'uniform_system' not in settings:
        # settings['uniform_system'] = True
        settings['uniform_system'] = False
    if 'plasma_dimension' not in settings:
        settings['plasma_dimension'] = 1.0
        # settings['plasma_dimension'] = 3.0
    if 'adaptive_dimension' not in settings:
        settings['adaptive_dimension'] = False
        # settings['adaptive_dimension'] = True
    if 'transition_type' not in settings:
        # settings['transition_type'] = 'none'
        # settings['transition_type'] = 'smooth_transition_to_uniform'
        settings['transition_type'] = 'smooth_transition_to_tR'
        # settings['transition_type'] = 'sharp_transition_to_tR'
    if 'adaptive_mirror' not in settings:
        # settings['adaptive_mirror'] = 'none'
        # settings['adaptive_mirror'] = 'adjust_U'
        # settings['adaptive_mirror'] = 'adjust_cell_size_with_mfp'
        settings['adaptive_mirror'] = 'adjust_cell_size_with_vth'
    if 'alpha_definition' not in settings:
        # settings['alpha_definition'] = 'old_constant'
        # settings['alpha_definition'] = 'geometric_constant'
        # settings['alpha_definition'] = 'geometric_constant_U0'
        settings['alpha_definition'] = 'geometric_local'
    if 'initialization_type' not in settings:
        # settings['initialization_type'] = 'linear_uniform'
        # settings['initialization_type'] = 'linear_alpha'
        settings['initialization_type'] = 'FD_decay'
    if 'left_boundary_condition' not in settings:
        settings['left_boundary_condition'] = 'enforce_tR'
        # settings['left_boundary_condition'] = 'uniform_scaling'
    if 'right_boundary_condition' not in settings:
        # settings['right_boundary_condition'] = 'enforce_tL'
        settings['right_boundary_condition'] = 'uniform_scaling'
    if 'right_boundary_condition_density_type' not in settings:
        settings['right_boundary_condition_density_type'] = 'n_transition'
        # settings['right_boundary_condition_density_type'] = 'n_expander'
    if 'ion_velocity_factor' not in settings:
        settings['ion_velocity_factor'] = 1.0
        # settings['ion_velocity_factor'] = np.sqrt(2)
    if 'electron_velocity_factor' not in settings:
        settings['electron_velocity_factor'] = 1.0
    if 'ion_scattering_rate_factor' not in settings:
        settings['ion_scattering_rate_factor'] = 1.0
        # settings['ion_scattering_rate_factor'] = 10.0
    if 'electron_scattering_rate_factor' not in settings:
        settings['electron_scattering_rate_factor'] = 1.0

    ### relaxation solver parameters
    if 't_stop' not in settings:
        # settings['t_stop'] = 1e-6
        # settings['t_stop'] = 1e-4
        # settings['t_stop'] = 1e-3
        # settings['t_stop'] = 1e-2
        # settings['t_stop'] = 3e-2
        settings['t_stop'] = 1e-1
        # settings['t_stop'] = 1.0
    if 't_solve_min' not in settings:
        settings['t_solve_min'] = 1e-20
        # settings['t_solve_min'] = 0.005
    if 'dt_status' not in settings:
        # settings['dt_status'] = 1e-7
        # settings['dt_status'] = 1e-6
        # settings['dt_status'] = 6e-6
        # settings['dt_status'] = 1e-5
        # settings['dt_status'] = 1e-4
        # settings['dt_status'] = 5e-4
        settings['dt_status'] = 1e-3
    if 'dt_factor' not in settings:
        settings['dt_factor'] = 0.3
    if 'dt_min' not in settings:
        settings['dt_min'] = 1e-20
    if 'n_min' not in settings:
        # settings['n_min'] = 0
        # settings['n_min'] = 1e10
        settings['n_min'] = 1e15
        # settings['n_min'] = 1e17
        # settings['n_min'] = 1e19
    if 'fail_on_minimal_density' not in settings:
        settings['fail_on_minimal_density'] = False
        # settings['fail_on_minimal_density'] = True
    if 'flux_normalized_termination_cutoff' not in settings:
        # settings['flux_normalized_termination_cutoff'] = 0.01
        settings['flux_normalized_termination_cutoff'] = 0.05
        # settings['flux_normalized_termination_cutoff'] = 0.3
    if 'print_time_step_info' not in settings:
        settings['print_time_step_info'] = True
    if 'do_plot_status' not in settings:
        settings['do_plot_status'] = True
    if 'save_state' not in settings:
        # settings['save_state'] = False
        settings['save_state'] = True
    if 'save_format' not in settings:
        settings['save_format'] = 'pickle'
        # settings['save_format'] = 'mat'
    if 'save_dir' not in settings:
        # settings['save_dir'] = 'runs/test/'
        # settings['save_dir'] = 'runs/test_U_0_smooth_transition/'
        # settings['save_dir'] = 'runs/test_U_1e5_smooth_transition/'
        # settings['save_dir'] = 'runs/test_U_1e6_smooth_transition/'
        # settings['save_dir'] = 'runs/test_U_7e5_smooth_transition/'
        # settings['save_dir'] = 'runs/test_U_0_smooth_transition_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_U_1e5_smooth_transition_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_U_1e6_smooth_transition_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_U_7e5_smooth_transition_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_U_' + '{:.0e}'.format(settings['U0']) + '_smooth_transition_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_smooth_transition/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_no_transition_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_no_transition_adjust_cell_size_right_bc_uniform_scaling/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_smooth_transition_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_smooth_transition_except_drag_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_sharp_transition_adjust_cell_size/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_smooth_transition_adjust_cell_size_right_bc_uniform_scaling/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_smooth_transition_adjust_cell_size_right_bc_uniform_scaling2/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_smooth_transition_adjust_cell_size_both_bc_uniform_scaling/'
        # settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0']) + '_transition_density_factor_0.5'
        settings['save_dir'] = 'runs/test_Rm_' + str(settings['Rm']) + '_U_' + '{:.1e}'.format(settings['U0'])
    if 'state_file' not in settings:
        settings['state_file'] = 'state'
    if 'settings_file' not in settings:
        settings['settings_file'] = 'settings'
    if 'log_file' not in settings:
        settings['log_file'] = 'log_file'
    if 'save_plots' not in settings:
        settings['save_plots'] = True
        # settings['save_plots'] = False
    if 'plots_x_axis' not in settings:
        # settings['plots_x_axis'] = 'total_length'
        settings['plots_x_axis'] = 'cell_number'

    return settings
