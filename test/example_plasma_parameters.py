import numpy as np
from matplotlib import pyplot as plt

from mm_rate_eqs.default_settings import define_default_settings
from mm_rate_eqs.fusion_functions import get_lawson_parameters, get_fusion_power, get_fusion_charged_power, \
    get_sigma_v_fusion
from mm_rate_eqs.plasma_functions import get_brem_radiation_loss, get_cyclotron_radiation_loss, get_magnetic_pressure, \
    get_ideal_gas_pressure, get_ideal_gas_energy_per_volume, get_magnetic_field_for_given_pressure, \
    get_bohm_diffusion_constant, get_larmor_radius, get_alfven_wave_group_velocity, get_larmor_frequency, \
    get_electron_plasma_frequency, get_ion_plasma_frequency
from mm_rate_eqs.rate_functions import calculate_coulomb_logarithm, get_thermal_velocity, get_coulomb_scattering_rate

from mm_rate_eqs.constants_functions import define_electron_mass, define_proton_mass, define_factor_eV_to_K, \
    define_boltzmann_constant, define_factor_Pa_to_bar

# lab plasma
# settings = {'gas_name': 'potassium'}
# T = 0.2
# # T = 2.0
# n_list = [1e15, 1e16, 1e17, 1e18]

# fusion plasma (linear)
# settings = {'gas_name': 'DT_mix'}
# T = 3500
# n_list = [1e22]

# settings = {'gas_name': 'DT_mix'}
# T = 3000
# T = 26000
# n_list = [2e21]
# n_list = [4e22]
# n_list = [8e22]
# n_list = [3.875e22]
# n_list = [3.875e22]

# fusion plasma
settings = {'gas_name': 'DT_mix'}
# settings = {'gas_name': 'hydrogen'}
# settings = {'gas_name': 'tritium'}
# T = 10000.0
# n_list = [2e20]
T = 10000.0
# T = 25000.0
n_list = [2e21]
# T = 26000.0
# n_list = [2e20]
# T = 1e3 * 100
# n_list = [2e20]
# T = 3000
# n_list = [4e22]
# n_list = [2e21]

# GOL-NB parameters
# settings = {'gas_name': 'hydrogen'}
# settings = {'gas_name': 'helium'}
# T = 40
# n_list = [3e19]
# settings['cell_size'] = 0.22

# T = 50000.0
# n_list = [2e21]

# fusion plasma
# settings = {'gas_name': 'DT_mix'}
# settings['diameter_main_cell'] = 0.2  # m
# T = 9000
# n_list = [5e21]

B = 1  # T
# B = 0.35  # T
# B = 3.5 #T
# B = 4.0  # T
# B = 5.0  # T
# B = 7.0 #T
# B = 10.0  # T

# fusion plasma (ITER)
# settings = {'gas_name': 'DT_mix'}
# T = 9000
# n_list = [1e20]
# B = 5.3 #T

# fusion plasma (DEMO)
# settings = {'gas_name': 'DT_mix'}
# T = 12900
# n_list = [8.7e19]
# B = 5.6  # T

settings['length_main_cell'] = 100  # m
settings['diameter_main_cell'] = 1  # m

settings = define_default_settings(settings=settings)
# settings['volume_main_cell'] = 1.0

Ti = T
Te = T
# Te = T / 2.0
Ti_keV = Ti / 1e3
Te_keV = Te / 1e3

for n in n_list:
    print('####################')
    print('n = ' + str(n) + ' m^-3')
    ne = n / 2
    ni = n / 2
    print('ni = ' + str(ni) + ' m^-3')
    print('Ti = ' + str(Ti) + ' eV = ' + str(Ti_keV) + ' keV')
    print('Coulomb log = ' + str(calculate_coulomb_logarithm(ne, Te, ni, Ti)['ii']))
    # scat_rate = get_specific_coulomb_scattering_rate(ne, Te, ni, Ti, settings, impact_specie='i', target_specie='i')
    scat_rate = get_coulomb_scattering_rate(ni, Ti, Te, settings, species='ions')
    tau_scat = 1 / scat_rate
    flux_scat = 0.5 * ni * settings['volume_main_cell'] / tau_scat

    print('ii scattering rate = ' + str(scat_rate) + ' [1/s]')
    # belan_scat_rate = get_coulomb_scattering_rate(n, T, T, settings, species='ions')
    # print('ii belan_scat_rate = ' + str(belan_scat_rate) + ' 1/s')
    v_th = get_thermal_velocity(Ti, settings, species='ions')
    print('v_th = ' + str(v_th) + ' m/s')
    mfp = v_th / scat_rate
    print('mfp = ' + str(mfp) + ' m')
    print('mfp = ' + str(mfp * 1e2) + ' cm')
    print('l = ' + str(settings['cell_size']) + ' m')
    print('mfp / l = ' + str(mfp / settings['cell_size']))

    # comparing scattering rate to referee #2 new model suggestion
    settings['Rm'] = 10.0
    alpha_approx = 1 / (4 * settings['Rm'])
    # from mm_rate_eqs.loss_cone_functions import get_solid_angles
    # alpha_tR, alpha_tL, alpha_c = get_solid_angles(0, v_th, 1 / settings['Rm'])
    # scat_rate_term_original = alpha_approx * scat_rate
    # scat_rate_term_referee2 = np.sqrt(scat_rate * v_th / settings['cell_size'])
    # print('scat_rate_term_original = ' + str(scat_rate_term_original) + ' [1/s]')
    # print('scat_rate_term_referee2 = ' + str(scat_rate_term_referee2) + ' [1/s]')
    # print(
    #     'scat_rate_term_referee2 / scat_rate_term_original = ' + str(scat_rate_term_referee2 / scat_rate_term_original))

    # Plasma beta
    P_magnetic = get_magnetic_pressure(B)
    P_plasma = get_ideal_gas_pressure(ne, Te, settings) + get_ideal_gas_pressure(ni, Ti, settings)
    print('B = ' + str(B) + ' T')
    beta = P_plasma / P_magnetic
    print('P_magnetic = ' + str(P_magnetic) + ' bar')
    print('P_plasma = ' + str(P_plasma) + ' bar')
    print('beta = ' + str(beta))
    B_for_given_P = get_magnetic_field_for_given_pressure(P_plasma, beta=1.0)  # [Tesla]
    print('B (for beta=1) = ' + str(B_for_given_P) + ' T')
    B_for_given_P = get_magnetic_field_for_given_pressure(P_plasma, beta=0.5)  # [Tesla]
    print('B (for beta=0.5) = ' + str(B_for_given_P) + ' T')

    # Summary of Lawson criterion
    print('mirror length = ' + str(settings['length_main_cell']) + ' m')
    print('mirror diameter = ' + str(settings['diameter_main_cell']) + ' m')
    print('mirror cross section = ' + str(settings['cross_section_main_cell']) + ' m^2')
    tau_lawson, flux_lawson = get_lawson_parameters(ni, Ti_keV, settings)
    print('tau_lawson: ', '{:.3e}'.format(tau_lawson), 's')
    print('tau_scat: ', '{:.3e}'.format(tau_scat), 's')
    print('ni * tau_lawson: ', '{:.3e}'.format(ni * tau_lawson))
    print('flux_lawson: ', '{:.3e}'.format(flux_lawson), 's^-1')

    print('flux_scat: ', '{:.3e}'.format(flux_scat), 's^-1')
    print('flux_scat / flux_lawson: ', '{:.3e}'.format(flux_scat / flux_lawson))

    # flux_single_mirror = 2 * v_th * n * settings['cross_section_main_cell']
    # print('flux single mirror: ', '{:.3e}'.format(flux_single_mirror), 's^-1')
    # print('flux_single_mirror / flux_lawson: ', '{:.3e}'.format(flux_single_mirror / flux_lawson))

    flux_naive = v_th * ni * settings['cross_section_main_cell']
    print('flux_naive: ', '{:.3e}'.format(flux_naive), 's^-1')
    print('flux_naive / flux_lawson: ', '{:.3e}'.format(flux_naive / flux_lawson))

    # Fusion power in nominal parameters
    # print('Main cell volume: ', '{:.3e}'.format(settings['volume_main_cell']), 'm^3')
    print('Main cell volume = ' + str(settings['volume_main_cell']) + ' m^3')
    sigma_v_fusion = get_sigma_v_fusion(Ti_keV)  # units m^3/s
    fusion_rate = 0.25 * ni * sigma_v_fusion  # scattering rate before multiplication by ni (so the units are 1/s)
    print('fusion_rate = ' + str(fusion_rate) + ' [1/s]')
    print('scat_rate / fusion_rate = ' + str(scat_rate / fusion_rate))
    Q_factor_collisions = 17600 * (1 / (1 + scat_rate / fusion_rate)) / (2 * Ti_keV)
    print('Q_factor_collisions = ' + str(Q_factor_collisions))

    P_fusion_volumetric = get_fusion_power(ni, Ti_keV)
    P_fusion = P_fusion_volumetric * settings['volume_main_cell']  # Watt
    print('P_fusion = ' + str(P_fusion / 1e6) + ' MW')
    P_fusion_charged_volumetric = get_fusion_charged_power(ni, Ti_keV)
    P_fusion_charged = P_fusion_charged_volumetric * settings['volume_main_cell']  # Watt
    print('P_fusion_charged = ' + str(P_fusion_charged / 1e6) + ' MW')

    # Radiation and Fusion, assuming Ti=3*Te
    P_brem_radiation_loss_volumetric = get_brem_radiation_loss(ni, ne, Te_keV, settings['Z_ion'])  # W/m^3
    P_cycl_radiation_loss_volumetric = get_cyclotron_radiation_loss(ne, Te_keV, B)  # W/m^3
    P_total_radiation_loss_volumetric = P_brem_radiation_loss_volumetric + P_cycl_radiation_loss_volumetric
    P_brem_radiation_loss = P_brem_radiation_loss_volumetric * settings['volume_main_cell']  # Watt
    P_cycl_radiation_loss = P_cycl_radiation_loss_volumetric * settings['volume_main_cell']  # Watt
    P_total_radiation_loss = P_total_radiation_loss_volumetric * settings['volume_main_cell']  # Watt
    print('P_brem_radiation_loss = ' + str(P_brem_radiation_loss / 1e6) + ' MW')
    print('P_cycl_radiation_loss = ' + str(P_cycl_radiation_loss / 1e6) + ' MW')
    print('P_total_radiation_loss = ' + str(P_total_radiation_loss / 1e6) + ' MW')

    # useful_energy_fraction = (P_fusion - P_total_radiation_loss) / P_fusion
    # print('useful_energy_fraction = ' + str(useful_energy_fraction))
    # net_useful_power = P_fusion - P_total_radiation_loss
    # print('net_useful_power = ' + str(net_useful_power / 1e6) + ' MW')
    Q_factor_rad = P_fusion / P_total_radiation_loss
    print('Q_factor_rad = ' + str(Q_factor_rad))

    # tau_confinement = 1e3  # sec
    tau_confinement = 30  # sec
    # tau_confinement = 1  # sec
    # tau_confinement = 0.5  # sec
    # tau_confinement = 6  # sec
    # tau_confinement = 1e-2  # sec
    print('tau_confinement = ' + str(tau_confinement) + ' s')
    E0_per_vol = get_ideal_gas_energy_per_volume(ne, Te, settings) \
                 + get_ideal_gas_energy_per_volume(ni, Ti, settings)  # energy per m^3
    print('E0_per_vol = ' + str(E0_per_vol) + ' J/m^3')
    E0_total = E0_per_vol * settings['volume_main_cell']  # J
    P_confinement_loss = E0_total / tau_confinement
    print('P_confinement_loss = ' + str(P_confinement_loss / 1e6) + ' MW')
    print('P_fus_normalized / P_rad = ' + str(P_fusion / P_total_radiation_loss))
    print('P_fus_normalized / P_conf = ' + str(P_fusion / P_confinement_loss))
    Q_factor_total = P_fusion / (P_confinement_loss + P_total_radiation_loss)
    print('Q_factor_total = ' + str(Q_factor_total))

    Q_factor_total_2 = (P_fusion - P_fusion_charged) / (
            P_confinement_loss + max(0, P_total_radiation_loss - P_fusion_charged))
    print('Q_factor_total_2 = ' + str(Q_factor_total_2))

    # Figuring out the radial diffusion flux
    print('Radial fluxes in the MM section:')

    # using Bohm diffusion
    print('Bohm diffusion')
    D_bohm = get_bohm_diffusion_constant(Te, B)  # [m^2/s]
    dndx = ne / (settings['diameter_main_cell'] / 2)
    radial_flux_density = D_bohm * dndx
    system_total_length = settings['length_main_cell']
    cyllinder_radial_cross_section = np.pi * settings['diameter_main_cell'] * system_total_length
    radial_flux = radial_flux_density * cyllinder_radial_cross_section
    print('radial_flux: ', '{:.3e}'.format(radial_flux), 's^-1')
    print('radial_flux / flux_naive: ', '{:.3e}'.format(radial_flux / flux_naive))

    # using classical diffusion
    print('classical diffusion')
    gyro_radius = get_larmor_radius(Ti, B)
    print('gyro_radius: ', '{:.3e}'.format(gyro_radius), 'm')
    D_classical = gyro_radius ** 2 * scat_rate
    radial_flux_density = D_classical * dndx
    radial_flux = radial_flux_density * cyllinder_radial_cross_section
    print('radial_flux: ', '{:.3e}'.format(radial_flux), 's^-1')
    print('radial_flux / flux_naive: ', '{:.3e}'.format(radial_flux / flux_naive))

    # Alfven waves
    v_alfven = get_alfven_wave_group_velocity(B, ni, gas_name=settings['gas_name'])
    print('v_alfven = ', '{:.3e}'.format(v_alfven), 'm/s')
    print('v_alfven / v_th: ', '{:.3e}'.format(v_alfven / v_th))

    # plasma frequency
    omega_elec_plasma_freq = 2 * np.pi * get_electron_plasma_frequency(ne)
    omega_ion_plasma_freq = 2 * np.pi * get_ion_plasma_frequency(ni, gas_name=settings['gas_name'])
    print('omega_elec_plasma_freq = ', '{:.3e}'.format(omega_elec_plasma_freq), '1/s')
    print('omega_ion_plasma_freq = ', '{:.3e}'.format(omega_ion_plasma_freq), '1/s')

    # cyclotron frequency
    f_cyclotron = get_larmor_frequency(B, settings['gas_name'])
    print('f_cyclotron = ', '{:.3e}'.format(f_cyclotron), '1/s')  # RF spans kHz to Ghz range, MHz in the middle
    omega_cyclotron = 2 * np.pi * f_cyclotron
    print('omega_cyclotron = ', '{:.3e}'.format(omega_cyclotron), '1/s')

    # comparing
    # TODO: make more detailed comparison of the actual legal modes
    print('v_exp = omega_cyclotron / k_paper ~= ', '{:.3e}'.format(omega_cyclotron / 1.0), 'm/s')
    print('v_th = ', '{:.3e}'.format(v_th), 'm/s')
    print('k_alfven ~ omega_cyclotron / v_alfven = ', '{:.3e}'.format(omega_cyclotron / v_alfven))
    # k_Lmode =

    # RF parameters
    # alpha_RF = 1.00001
    # vz_res = 1.0 * v_th
    alpha_RF = 0.9
    vz_res = 0.5 * v_th
    # alpha_RF = 1.03
    # vz_res = 1.5 * v_th
    omega_RF = alpha_RF * omega_cyclotron
    if alpha_RF != 1.0:
        v_RF = vz_res * alpha_RF / (alpha_RF - 1.0)
    else:
        v_RF = vz_res
    f_RF = omega_RF / (2 * np.pi)
    lambda_RF = v_RF / f_RF
    print('lambda_RF = ', '{:.3e}'.format(lambda_RF), 'm')
    print('lambda_RF / gyro_radius = ', '{:.3e}'.format(lambda_RF / gyro_radius))

    # relative strength of magnetic and electric fields in Maxwell consistent RF field
    r = 1  # m (typical particle distance from mirror axis)
    c = 3e8  # m/s (speed of light)
    relativistic_error_dimensionless = (r * omega_cyclotron / c) ** 2
    print('relativistic_error_dimensionless = ', '{:.3e}'.format(relativistic_error_dimensionless), )

    E_RF = 10 * 1e3  # V/m
    B_RF = E_RF * r * omega_cyclotron / c ** 2  # in [Tesla]=[V*s/m^2]
    print('input E=' + '{:.3e}'.format(E_RF / 1e3) + 'kV/m gives B=' + '{:.3e}'.format(B_RF) + 'T')
    F_magnetic_over_electric = B_RF * v_th / E_RF  # forces comparison
    # print('Force ratio vB/E = ' + '{:.3e}'.format(F_magnetic_over_electric))
    print('Force ratio E/vB = ' + '{:.3e}'.format(1 / F_magnetic_over_electric))

    # B_RF = 1e-3 # T
    B_RF = 3e-4  # T
    E_RF = B_RF * r * omega_cyclotron  # in [V/m]
    print('input B=' + '{:.3e}'.format(B_RF) + 'T, gives E=' + '{:.3e}'.format(E_RF / 1e3) + 'kV/m')
    F_magnetic_over_electric = B_RF * v_th / E_RF  # forces comparison
    print('Force ratio vB/E = ' + '{:.3e}'.format(F_magnetic_over_electric))
    # F_electric_over_magnetic = B_RF * v_th / E_RF # forces comparison
    # print('Force ratio vB/E = ' + '{:.3e}'.format(F_magnetic_over_electric))
