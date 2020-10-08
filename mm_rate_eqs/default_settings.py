import numpy as np

from mm_rate_eqs.fusion_functions import define_plasma_parameters


def define_default_settings(settings=None):
    if settings == None:
        settings = {}

    #### physical constants
    settings['eV'] = 1.0
    settings['keV'] = 1e3 * settings['eV']
    settings['eV_to_K'] = 1.16e4
    settings['MeV_to_J'] = 1e6 * 1.6e-19
    settings['kB_K'] = 1.380649e-23  # J/K
    settings['e'] = 1.60217662e-19  # Coulomb (elementary charge)
    settings['kB_eV'] = settings['kB_K'] * settings['eV_to_K']  # J/eV (numerically same as e)
    settings['eps0'] = 8.85418781e-12  # Farad/m^2 (vacuum permittivity)

    ### plasma parameters5
    if 'gas_name' not in settings:
        settings['gas_name'] = 'hydrogen'
    if 'ionization_level' not in settings:
        settings['ionization_level'] = 1.0
        # settings['ionization_level'] = None
    settings['me'], settings['mp'], settings['mi'], settings['A_atomic_weight'], settings['Z_ion'] \
        = define_plasma_parameters(gas_name=settings['gas_name'], ionization_level=settings['ionization_level'])
    settings['coulomb_log_min'] = 1.0

    ### system parameters
    if 'n0' not in settings:
        # settings['n0'] = 10e22  # m^-3
        # settings['n0'] = 3.875e22  # m^-3
        settings['n0'] = 2e22  # m^-3
        # settings['n0'] = 1e22  # m^-3
        # settings['n0'] = 5e21  # m^-3
    if 'Ti_0' not in settings:
        settings['Ti_0'] = 3 * settings['keV']
        # settings['Ti_0'] = 9 * settings['keV']
    if 'Te_0' not in settings:
        # settings['Te_0'] = 1 * settings['keV']
        settings['Te_0'] = 3 * settings['keV']
        # settings['Te_0'] = 9 * settings['keV']
    if 'Rm' not in settings:
        # settings['Rm'] = 1.4
        # settings['Rm'] = 2.0
        settings['Rm'] = 3.0
        # settings['Rm'] = 5.0
    if 'U0' not in settings:
        settings['U0'] = 0
        # settings['U0'] = 0.001
        # settings['U0'] = 0.01
        # settings['U0'] = 0.1
    if 'mmm_velocity_type' not in settings:
        # settings['mmm_velocity_type'] = 'absolute'
        settings['mmm_velocity_type'] = 'relative_to_thermal_velocity'
    if 'cell_size' not in settings:
        # settings['cell_size'] = 3.0  # m (MMM wavelength)
        # settings['cell_size'] = 4.0  # m (MMM wavelength)
        # settings['cell_size'] = 5.0  # m (MMM wavelength)
        settings['cell_size'] = 10.0  # m (MMM wavelength)
    if 'mfp_min' not in settings:
        # settings['mfp_min'] = 0.1 * settings['cell_size']
        settings['mfp_min'] = 0.01 * settings['cell_size']
    if 'mfp_max' not in settings:
        settings['mfp_max'] = 100 * settings['cell_size']
        # settings['mfp_max'] = 50 * settings['cell_size']
        # settings['mfp_max'] = 10 * settings['cell_size']
    if 'delta_n_smoothing_factor' not in settings:
        settings['delta_n_smoothing_factor'] = 0.01
        # settings['delta_n_smoothing_factor'] = 0.05
        # settings['delta_n_smoothing_factor'] = 0.1
        # settings['delta_n_smoothing_factor'] = 0.5
    if 'number_of_cells' not in settings:
        settings['number_of_cells'] = 10
        # settings['number_of_cells'] = 50
        # settings['number_of_cells'] = 100
        # settings['number_of_cells'] = 150
        # settings['number_of_cells'] = 200
        # settings['number_of_cells'] = 300
        # settings['number_of_cells'] = 1000
    if 'length_main_cell' not in settings:
        settings['length_main_cell'] = 100  # m
    if 'diameter_main_cell' not in settings:
        settings['diameter_main_cell'] = 0.5  # m
    settings['cross_section_main_cell'] = np.pi * (settings['diameter_main_cell'] / 2) ** 2  # m^3
    settings['volume_main_cell'] = settings['length_main_cell'] * settings['cross_section_main_cell']  # m^3

    ### additional options
    if 'assume_constant_density' not in settings:
        # settings['assume_constant_density'] = True
        settings['assume_constant_density'] = False
    if 'assume_constant_temperature' not in settings:
        # settings['assume_constant_temperature'] = True
        settings['assume_constant_temperature'] = False
    if 'assume_constant_transmission' not in settings:
        # settings['assume_constant_transmission'] = True
        settings['assume_constant_transmission'] = False
    if 'energy_conservation_scheme' not in settings:
        settings['energy_conservation_scheme'] = 'none'
        # settings['energy_conservation_scheme'] = 'simple'
        # settings['energy_conservation_scheme'] = 'detailed'
    if 'plasma_dimension' not in settings:
        settings['plasma_dimension'] = 1.0
        # settings['plasma_dimension'] = 1.5
        # settings['plasma_dimension'] = 1.8
        # settings['plasma_dimension'] = 2.0
        # settings['plasma_dimension'] = 2.5
        # settings['plasma_dimension'] = 3.0
    if 'adaptive_dimension' not in settings:
        settings['adaptive_dimension'] = False
        # settings['adaptive_dimension'] = True
    if 'transition_type' not in settings:
        settings['transition_type'] = 'none'
        # settings['transition_type'] = 'smooth_transition_to_free_flow'
        # settings['transition_type'] = 'sharp_transition_to_free_flow'
    if 'adaptive_mirror' not in settings:
        settings['adaptive_mirror'] = 'none'
        # settings['adaptive_mirror'] = 'adjust_U'
        # settings['adaptive_mirror'] = 'adjust_cell_size_with_mfp'
        # settings['adaptive_mirror'] = 'adjust_cell_size_with_vth'
    if 'alpha_definition' not in settings:
        # settings['alpha_definition'] = 'old_constant'
        # settings['alpha_definition'] = 'geometric_constant'
        # settings['alpha_definition'] = 'geometric_constant_U0'
        settings['alpha_definition'] = 'geometric_local'
    if 'U_for_loss_cone_factor' not in settings:
        settings['U_for_loss_cone_factor'] = 1.0
    if 'initialization_type' not in settings:
        # settings['initialization_type'] = 'linear_uniform'
        # settings['initialization_type'] = 'linear_alpha'
        settings['initialization_type'] = 'FD_decay'
    if 'left_boundary_condition' not in settings:
        settings['left_boundary_condition'] = 'adjust_ntR_for_n0'
        # settings['left_boundary_condition'] = 'adjust_all_species_for_n0'
    if 'right_boundary_condition' not in settings:
        settings['right_boundary_condition'] = 'none'
        # settings['right_boundary_condition'] = 'adjust_ntL_for_nend'
        # settings['right_boundary_condition'] = 'adjust_all_species_for_nend'
        # settings['right_boundary_condition'] = 'nullify_ntL'
    if 'right_boundary_condition_density_type' not in settings:
        settings['right_boundary_condition_density_type'] = 'none'
        # settings['right_boundary_condition_density_type'] = 'n_transition'
        # settings['right_boundary_condition_density_type'] = 'n_expander'
    if 'n_expander_factor' not in settings:
        settings['n_expander_factor'] = 0.01
    if 'nullify_ntL_factor' not in settings:
        settings['nullify_ntL_factor'] = 0.05
    if 'transmission_factor' not in settings:
        # diffusion of ions increased by a factor of (Ti + Te)/Ti, see Bellan p. 19
        settings['transmission_factor'] = (settings['Ti_0'] + settings['Te_0']) / settings['Ti_0']
        # settings['transmission_factor'] = 1.0
        # settings['transmission_factor'] = np.sqrt(2)
    if 'ion_scattering_rate_factor' not in settings:
        settings['ion_scattering_rate_factor'] = 1.0
        # settings['ion_scattering_rate_factor'] = 30.0
        # settings['ion_scattering_rate_factor'] = 100.0
    if 'electron_scattering_rate_factor' not in settings:
        settings['electron_scattering_rate_factor'] = 1.0

    ### relaxation solver parameters
    if 'max_num_time_steps' not in settings:
        settings['max_num_time_steps'] = int(3e5) - 1
        # settings['max_num_time_steps'] = int(status_counter_type1e5) - 1
        # settings['max_num_time_steps'] = int(1e10) - 1
    if 't_stop' not in settings:
        # settings['t_stop'] = 1e-6
        # settings['t_stop'] = 1e-3
        # settings['t_stop'] = 1e-2
        # settings['t_stop'] = 3e-2
        settings['t_stop'] = 1e-1
        # settings['t_stop'] = 1.0
    if 't_solve_min' not in settings:
        settings['t_solve_min'] = 1e-20
        # settings['t_solve_min'] = 0.005
    if 'status_counter_type' not in settings:
        # settings['status_counter_type'] = 'time_elapsed'
        settings['status_counter_type'] = 'time_steps'
    if 'dt_status' not in settings:
        # settings['dt_status'] = 1e-7
        # settings['dt_status'] = 1e-6
        # settings['dt_status'] = 6e-6
        # settings['dt_status'] = 1e-5
        settings['dt_status'] = 1e-4
        # settings['dt_status'] = 5e-4
        # settings['dt_status'] = 1e-3
    if 'time_steps_status' not in settings:
        settings['time_steps_status'] = int(1e3)
        # settings['time_steps_status'] = int(1e4)
    if 'dt_factor' not in settings:
        # settings['dt_factor'] = 0.3
        settings['dt_factor'] = 0.1
    if 'dt_min' not in settings:
        settings['dt_min'] = 1e-20
    if 'time_step_definition_using_species' not in settings:
        settings['time_step_definition_using_species'] = 'all'
        # settings['time_step_definition_using_species'] = 'only_c_tR'

    if 'n_min' not in settings:  # minimal density allowed in the entire system
        # settings['n_min'] = 0
        settings['n_min'] = 1e10
        # settings['n_min'] = 1e15
        # settings['n_min'] = 1e17
        # settings['n_min'] = 1e19
    if 'n_end_min' not in settings:  # minimal density allowed as the right boundary condition and n_transition
        settings['n_end_min'] = 1e18
        # settings['n_end_min'] = 1e20
        # settings['n_end_min'] = 5e20
    if 'fail_on_minimal_density' not in settings:
        settings['fail_on_minimal_density'] = False
        # settings['fail_on_minimal_density'] = True

    if 'flux_normalized_termination_cutoff' not in settings:
        # settings['flux_normalized_termination_cutoff'] = 0.01
        settings['flux_normalized_termination_cutoff'] = 0.05
        # settings['flux_normalized_termination_cutoff'] = 0.1
        # settings['flux_normalized_termination_cutoff'] = 0.3
    if 'print_time_step_info' not in settings:
        settings['print_time_step_info'] = True

    if 'draw_plots' not in settings:
        settings['draw_plots'] = True
    if 'save_plots_scheme' not in settings:
        # settings['save_plots_scheme'] = 'none'
        settings['save_plots_scheme'] = 'status_plots'
        # settings['save_plots_scheme'] = 'only_at_calculation_end'

    if 'save_state' not in settings:
        # settings['save_state'] = False
        settings['save_state'] = True
    if 'save_format' not in settings:
        settings['save_format'] = 'pickle'
        # settings['save_format'] = 'mat'
    if 'save_dir' not in settings:
        settings['save_dir'] = 'runs/test/'

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
