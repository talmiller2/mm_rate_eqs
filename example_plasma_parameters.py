from default_settings import define_default_settings
from fusion_functions import get_lawson_parameters, get_brem_radiation_loss, get_cyclotron_radiation_loss, \
    get_fusion_power, get_magnetic_pressure, get_ideal_gas_pressure, get_fusion_charged_power, \
    get_ideal_gas_energy_per_volume
from rate_functions import calculate_coulomb_logarithm, get_specific_coulomb_scattering_rate, get_thermal_velocity

# lab plasma
# settings = {'gas_name': 'potassium'}
# T = 0.2
# # T = 2.0
# n_list = [1e15, 1e16, 1e17, 1e18]

# fusion plasma (linear)
settings = {'gas_name': 'DT_mix'}
T = 3500
n_list = [1e22]

# fusion plasma
# settings = {'gas_name': 'DT_mix'}
# T = 10000
# n_list = [1e20]

# fusion plasma
# settings = {'gas_name': 'DT_mix'}
# settings['diameter_main_cell'] = 0.2  # m
# T = 9000
# n_list = [5e21]

#
# # B = 3.5 #T
B = 5.0  # T
# B = 7.0 #T

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
    print('Ti = ' + str(Ti) + ' eV = ' + str(Ti_keV) + ' keV')
    print('Coulomb log = ' + str(calculate_coulomb_logarithm(ne, Te, ni, Ti)['ii']))
    scat_rate = get_specific_coulomb_scattering_rate(ne, Te, ni, Ti, settings, impact_specie='i', target_specie='i')
    print('ii scattering rate = ' + str(scat_rate) + ' 1/s')
    # belan_scat_rate = get_coulomb_scattering_rate(n, T, T, settings, species='ions')
    # print('ii belan_scat_rate = ' + str(belan_scat_rate) + ' 1/s')
    v_th = get_thermal_velocity(Ti, settings, species='ions')
    print('v_th = ' + str(v_th) + ' m/s')
    mfp = v_th / scat_rate
    print('mfp = ' + str(mfp) + ' m')
    print('mfp = ' + str(mfp * 1e2) + ' cm')

    # Plasma beta
    P_magnetic = get_magnetic_pressure(B)
    P_plasma = get_ideal_gas_pressure(ne, Te, settings) + get_ideal_gas_pressure(ni, Ti, settings)
    beta = P_plasma / P_magnetic
    print('P_magnetic = ' + str(P_magnetic) + ' bar')
    print('P_plasma = ' + str(P_plasma) + ' bar')
    print('beta = ' + str(beta))

    # Summary of Lawson criterion
    tau_lawson, flux_lawson = get_lawson_parameters(ni, Ti, settings)
    print('tau_lawson: ', '{:.3e}'.format(tau_lawson), 's')
    print('flux_lawson: ', '{:.3e}'.format(flux_lawson), 's^-1')
    flux_single_mirror = 2 * v_th * n * settings['cross_section_main_cell']
    print('flux single mirror: ', '{:.3e}'.format(flux_single_mirror), 's^-1')
    print('flux_single_mirror / flux_lawson: ', '{:.3e}'.format(flux_single_mirror / flux_lawson))

    # Fusion power in nominal parameters
    # print('Main cell volume: ', '{:.3e}'.format(settings['volume_main_cell']), 'm^3')
    print('Main cell volume = ' + str(settings['volume_main_cell']) + ' m^3')
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
    print('P_fus / P_rad = ' + str(P_fusion / P_total_radiation_loss))
    print('P_fus / P_conf = ' + str(P_fusion / P_confinement_loss))
    Q_factor_total = P_fusion / (P_confinement_loss + P_total_radiation_loss)
    print('Q_factor_total = ' + str(Q_factor_total))

    Q_factor_total_2 = (P_fusion - P_fusion_charged) / (
                P_confinement_loss + max(0, P_total_radiation_loss - P_fusion_charged))
    print('Q_factor_total_2 = ' + str(Q_factor_total_2))
