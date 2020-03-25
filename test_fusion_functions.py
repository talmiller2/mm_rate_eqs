import matplotlib.pyplot as plt
import numpy as np

from default_settings import define_default_settings
from fusion_functions import get_sigma_v_fusion, get_E_reaction, get_brem_radiation_loss, get_cyclotron_radiation_loss, \
    get_fusion_power, get_lawson_parameters
from rate_functions import get_thermal_velocity

### Plot fusion and radiation loss parameters

settings = define_default_settings()
keV = settings['keV']
eV_to_K = settings['eV_to_K']
Z_charge = settings['Z_charge']
B = settings['B']
n0 = settings['n0']
Ti_0 = settings['Ti_0']
Te_0 = settings['Te_0']

T_keV_array = np.linspace(0.2, 200, 1000)
# reactions = ['D-T_to_n_alpha', 'D-D_to_p_T', 'D-D_to_n_He3', 'He3-D_to_p_alpha']
reactions = ['D-T_to_n_alpha', 'D-D_to_p_T_n_He3', 'He3-D_to_p_alpha', 'p_B_to_3alpha']
# reactions = ['D-T_to_n_alpha', 'D-D_to_p_T_n_He3']

plt.rcParams.update({'font.size': 16})
plt.close('all')

plt.figure()
for reaction in reactions:
    sigma_v_array = get_sigma_v_fusion(T_keV_array * 1e3, reaction=reaction)
    E_reaction = get_E_reaction(reaction=reaction)
    label = reaction + ', $E_{reaction}$=' + str(round(E_reaction, 3)) + 'MeV'
    plt.plot(T_keV_array, sigma_v_array, label=label, linewidth=3)
plt.legend()
plt.ylabel('$\\sigma*v [m^3/s]$')
plt.xlabel('T [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

plt.figure()
sigma_v_array_ref = get_sigma_v_fusion(T_keV_array * 1e3, reaction=reactions[0])
for reaction in reactions:
    sigma_v_array = get_sigma_v_fusion(T_keV_array * 1e3, reaction=reaction)
    E_reaction = get_E_reaction(reaction=reaction)
    label = reaction + ', $E_{reaction}$=' + str(round(E_reaction, 3)) + 'MeV'
    plt.plot(T_keV_array, sigma_v_array / sigma_v_array_ref, label=label, linewidth=3)
plt.legend()
plt.ylabel('$\\sigma*v$ relative to DT')
plt.xlabel('T [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

# Radiation and Fusion, assuming Ti=Te
P_brem_radiation_loss_volumetric = get_brem_radiation_loss(n0, n0, T_keV_array, Z_charge)  # W/m^3
P_cycl_radiation_loss_volumetric = get_cyclotron_radiation_loss(n0, T_keV_array, B)  # W/m^3
P_cycl_radiation_loss_volumetric_total = P_brem_radiation_loss_volumetric + P_cycl_radiation_loss_volumetric
plt.figure()
plt.plot(T_keV_array, P_brem_radiation_loss_volumetric, '--', label='brem loss', linewidth=3)
plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric, '--', label='cyclotron loss', linewidth=3)
plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric_total, '--', label='total loss', linewidth=3)
for reaction in reactions:
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric, label=reaction, linewidth=3)
plt.legend()
plt.title('Fusion and radiation loss power, $T_i=T_e$')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$W/m^3$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

plt.figure()
for reaction in reactions:
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric / P_cycl_radiation_loss_volumetric_total, label=reaction, linewidth=3)
plt.legend()
plt.title('Fusion to radiation loss ratio, $T_i=T_e$')
plt.xlabel('$T_i$ [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

# Radiation and Fusion, assuming Ti=3*Te
P_brem_radiation_loss_volumetric = get_brem_radiation_loss(n0, n0, T_keV_array / 3.0, Z_charge)  # W/m^3
P_cycl_radiation_loss_volumetric = get_cyclotron_radiation_loss(n0, T_keV_array / 3.0, B)  # W/m^3
P_cycl_radiation_loss_volumetric_total = P_brem_radiation_loss_volumetric + P_cycl_radiation_loss_volumetric
plt.figure()
plt.plot(T_keV_array, P_brem_radiation_loss_volumetric, '--', label='brem loss', linewidth=3)
plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric, '--', label='cyclotron loss', linewidth=3)
plt.plot(T_keV_array, P_cycl_radiation_loss_volumetric_total, '--', label='total loss', linewidth=3)
for reaction in reactions:
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric, label=reaction, linewidth=3)
plt.legend()
plt.title('Fusion and radiation loss power, $T_i=3T_e$')
plt.xlabel('$T_i$ [keV]')
plt.ylabel('$W/m^3$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

plt.figure()
for reaction in reactions:
    P_fusion_volumetric = get_fusion_power(n0, T_keV_array, reaction=reaction)
    plt.plot(T_keV_array, P_fusion_volumetric / P_cycl_radiation_loss_volumetric_total, label=reaction, linewidth=3)
plt.legend()
plt.title('Fusion to radiation loss ratio, $T_i=3T_e$')
plt.xlabel('$T_i$ [keV]')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.grid()

# Summary of Lawson criterion
tau_lawson, flux_lawson = get_lawson_parameters(n0, Ti_0, settings)
print('tau_lawson: ', '{:.3e}'.format(tau_lawson), 's')
print('flux_lawson: ', '{:.3e}'.format(flux_lawson), 's^-1')
v_th = get_thermal_velocity(Ti_0, settings)
flux_single_mirror = v_th * n0 * settings['cross_section_main_cell']
print('flux single mirror: ', '{:.3e}'.format(flux_single_mirror), 's^-1')

# Fusion power in nominal parameters
print('Main cell volume: ', '{:.3e}'.format(settings['volume_main_cell']), 'm^3')
P_fusion_volumetric = get_fusion_power(n0, Ti_0 / settings['keV'])
P_fusion = P_fusion_volumetric * settings['volume_main_cell']  # Watt
print('Fusion power: ', '{:.3e}'.format(P_fusion / 1e6), 'MW')
