import numpy as np
from matplotlib import pyplot as plt
from mm_rate_eqs.constants_functions import define_electron_charge, define_electron_mass, define_proton_mass, \
    define_vacuum_permittivity
from mm_rate_eqs.plasma_functions import get_debye_length
import copy

plt.rcParams['font.size'] = 12
plt.close('all')

# Plot the plasma density input profile
ne_max = 1e21
# ne_max = 5e18
r_wall = 0.3
# r_wall = 0.5
# r = np.linspace(0, 2 * r_wall, 100)
r = np.linspace(0, 1, 100)
# ne = ne_max * (1 - np.tanh(r / r_wall * np.pi))
dr_FD = r_wall / 5
# dr_FD = r_wall / 10
ne = ne_max / (1 + np.exp((r - r_wall) / dr_FD))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(r, ne / ne_max)
plt.xlabel('r [m]')
# plt.ylabel('ne [m^-3]')
plt.ylabel('ne / ne_max')
plt.grid(True)
# plt.tight_layout()

L = 3
dz_FD = L / 30
z_min = 0 + 5 * dz_FD
z_max = L - 5 * dz_FD
z = np.linspace(-1, 4, 100)
ne = ne_max / (1 + np.exp(-(z - z_min) / dz_FD)) / (1 + np.exp((z - z_max) / dz_FD))
# plt.figure()
plt.subplot(1, 2, 2)
plt.plot(z, ne / ne_max)
plt.xlabel('z [m]')
# plt.ylabel('ne [m^-3]')
plt.grid(True)
plt.tight_layout()

e_const = define_electron_charge()
me_const = define_electron_mass()
mp_const = define_proton_mass()
epsilon0_const = define_vacuum_permittivity()

# from mm_rate_eqs.plasma_functions import get_electron_plasma_frequency
# wpe_calc = get_electron_plasma_frequency(1e19)
# print(wpe_calc / 1e6)

A = 0.5  # Z/m for deuterium
# f_rf = 1e6 # [Hz]
# f_rf = 100e6 # [Hz]
# f_rf = 1e-8 # [Hz]
f_rf = np.logspace(4, 13, 1000)  # [Hz]
# f_rf = np.logspace(6, 8, 100) # [Hz]
omega_rf = 2 * np.pi * f_rf
Z_e = -1
Z_i = 1
Bz = 1  # [T]
eps = 1e-5
# Br = 0 # [T]
Br = eps  # [T]
Bphi = 0  # [T]
# Bphi = eps # [T]
B = np.sqrt(Bz ** 2 + Br ** 2 + Bphi ** 2)
# ne_max = 1e15 # [m^-3]
ne_max = 1e20  # [m^-3]
ne = ne_max
collision_space = 0
# collision_space = 0.001
wce = -e_const * B / me_const / (1 - 1j * collision_space)
wci = A * e_const * B / mp_const / (1 - 1j * collision_space)
wpe_sqr = ne * e_const ** 2 / (me_const * epsilon0_const) / (1 + 1j * collision_space)
wpi_sqr = A * ne * e_const ** 2 / (mp_const * epsilon0_const) / (1 + 1j * collision_space)
# print('wpe^2/wce=', wpe_sqr / wce)
# print('wpi^2/wci=', wpi_sqr / wci)
# print('wpe^2/wce / wpi^2/wci=', (wpe_sqr / wce) / (wpi_sqr / wci))
S = 1 - wpe_sqr / (omega_rf ** 2 - wce ** 2) \
    - wpi_sqr / (omega_rf ** 2 - wci ** 2)
D = wce * wpe_sqr / (omega_rf * (omega_rf ** 2 - wce ** 2)) \
    + wci * wpi_sqr / (omega_rf * (omega_rf ** 2 - wci ** 2))
P = 1 - wpe_sqr / omega_rf ** 2 - wpi_sqr / omega_rf ** 2
print('S=', S, 'D=', D, 'P=', P)
print('P/S=', P / S)
# R_stix = S + D
# L_stix = S - D

th = np.arccos(Bz / B)
ph = np.arctan2(Bz, Br)
ux = -Bphi / np.sqrt(Br ** 2 + Bphi ** 2)
uy = Br / np.sqrt(Br ** 2 + Bphi ** 2)
# print('ux=', ux, ', uy=', uy, ', cos(th)=', np.cos(th), ', sin(th)=', np.sin(th))
uz = 0
e_xx = S * ux ** 2 * (ux ** 2 + uy ** 2) - 2 * S * ux ** 2 * (-1 + ux ** 2 + uy ** 2) * np.cos(th) \
       + S * (1 + ux ** 4 + ux ** 2 * (-2 + uy ** 2)) * np.cos(th) ** 2 + P * uy ** 2 * np.sin(th) ** 2
e_xy = (ux * uy - ux * uy * np.cos(th)) * (
            S * (ux ** 2 + np.cos(th) - ux ** 2 * np.cos(th)) - 1j * D * (-(ux * uy) + ux * uy * np.cos(th))) \
       + (uy ** 2 + np.cos(th) - uy ** 2 * np.cos(th)) * (
                   -1j * D * (ux ** 2 + np.cos(th) - ux ** 2 * np.cos(th)) - S * (-(ux * uy) + ux * uy * np.cos(th))) \
       - P * ux * uy * np.sin(th) ** 2
e_xz = P * uy * np.cos(th) * np.sin(th) - uy * (
            S * (ux ** 2 + np.cos(th) - ux ** 2 * np.cos(th)) - 1j * D * (-(ux * uy) + ux * uy * np.cos(th))) * np.sin(
    th) + ux * (-1j * D * (ux ** 2 + np.cos(th) - ux ** 2 * np.cos(th)) - S * (
            -(ux * uy) + ux * uy * np.cos(th))) * np.sin(th)
e_yx = (ux ** 2 + np.cos(th) - ux ** 2 * np.cos(th)) * (
            S * (ux * uy - ux * uy * np.cos(th)) + 1j * D * (uy ** 2 + np.cos(th) - uy ** 2 * np.cos(th))) + (
                   ux * uy - ux * uy * np.cos(th)) * (-1j * D * (ux * uy - ux * uy * np.cos(th)) + S * (
            uy ** 2 + np.cos(th) - uy ** 2 * np.cos(th))) - P * ux * uy * np.sin(th) ** 2
e_yy = S * uy ** 2 * (ux ** 2 + uy ** 2) - 2 * S * uy ** 2 * (-1 + ux ** 2 + uy ** 2) * np.cos(th) + S * (
            1 + (-2 + ux ** 2) * uy ** 2 + uy ** 4) * np.cos(th) ** 2 + P * ux ** 2 * np.sin(th) ** 2
e_yz = - (P * ux * np.cos(th) * np.sin(th)) - uy * (
            S * (ux * uy - ux * uy * np.cos(th)) + 1j * D * (uy ** 2 + np.cos(th) - uy ** 2 * np.cos(th))) * np.sin(
    th) + ux * (-1j * D * (ux * uy - ux * uy * np.cos(th)) + S * (
            uy ** 2 + np.cos(th) - uy ** 2 * np.cos(th))) * np.sin(th)
e_zx = P * uy * np.cos(th) * np.sin(th) + (ux * uy - ux * uy * np.cos(th)) * (
            S * ux * np.sin(th) + 1j * D * uy * np.sin(th)) + (ux ** 2 + np.cos(th) - ux ** 2 * np.cos(th)) * (
                   1j * D * ux * np.sin(th) - S * uy * np.sin(th))
e_zy = -(P * ux * np.cos(th) * np.sin(th)) + (uy ** 2 + np.cos(th) - uy ** 2 * np.cos(th)) * (
            S * ux * np.sin(th) + 1j * D * uy * np.sin(th)) + (ux * uy - ux * uy * np.cos(th)) * (
                   1j * D * ux * np.sin(th) - S * uy * np.sin(th))
e_zz = P * np.cos(th) ** 2 + S * (ux ** 2 + uy ** 2) * np.sin(th) ** 2
# e_tensor = np.array([[e_xx, e_xy, e_xz], [e_yx, e_yy, e_yz], [e_zx, e_zy, e_zz]])

# # Plug Bz-only limit (where ux=uy=th=0)
# e_xx = S
# e_xy = -1j * D
# e_xz = 0
# e_yx = 1j * D
# e_yy = S
# e_yz = 0
# e_zx = 0
# e_zy = 0
# e_zz = P

e_tensor = np.array([[S, 1j * D, 0], [1j * D, S, 0], [0, 0, P]])

# print('e_tensor=', e_tensor)
# print('Re e_tensor=', np.real(e_tensor))
# print('Im e_tensor=', np.imag(e_tensor))

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(np.real(e_tensor))
# plt.title('np.real(e_tensor)')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(np.imag(e_tensor))
# plt.title('np.imag(e_tensor)')
# plt.colorbar()
# plt.tight_layout()
#
#
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(np.log10(abs(np.real(e_tensor))))
# plt.title('np.log10(abs(np.real(e_tensor)))')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(np.log10(abs(np.imag(e_tensor))))
# plt.title('np.log10(abs(np.imag(e_tensor)))')
# plt.colorbar()
# plt.tight_layout()


# plot components as a function of frequency
data_dict = {}
data_dict['S'] = S
data_dict['D'] = D
# data_dict['D1'] = D1
# data_dict['D2'] = D2
# data_dict['D/S'] = D / S
# data_dict['(D/S)^2'] = (D / S) ** 2
# data_dict['1-(D/S)^2'] = 1 - (D / S) ** 2
data_dict['S(1-(D/S)^2)'] = S * (1 - (D / S) ** 2)
data_dict['P'] = P
# data_dict['P/S_tilde'] = P / (S * (1 - (D / S) ** 2))

color_dict = {}
color_dict['S'] = 'b'
color_dict['D'] = 'g'
color_dict['D1'] = 'b'
color_dict['D2'] = 'r'
color_dict['D/S'] = 'orange'
color_dict['(D/S)^2'] = 'orange'
color_dict['1-(D/S)^2'] = 'orange'
color_dict['S(1-(D/S)^2)'] = 'm'
color_dict['P'] = 'r'
color_dict['P/S_tilde'] = 'r'

plt.figure()

# ymin = np.real(np.min([np.min(data_dict[key]) for key in data_dict]))
# ymax = np.real(np.max([np.max(data_dict[key]) for key in data_dict]))
ymin = 0
ymax = np.real(np.max([np.max(np.abs(data_dict[key])) for key in data_dict]))
x_interest = np.array([0.5, 1.5])
color_markers = 'grey'
# color_markers = 'yellow'
# plt.fill_between(x_interest, ymin, ymax, color=color_markers, alpha=0.3)
# y_text = ymax / 1e2
y_text = 1e9
plt.vlines(wci / wci, ymin, ymax, color=color_markers)
plt.text(wci / wci, y_text, '$\\omega_{ci}$')
plt.vlines(abs(wce) / wci, ymin, ymax, color=color_markers)
plt.text(abs(wce) / wci, y_text, '$\\omega_{ce}$')
plt.vlines(np.sqrt(wpi_sqr) / wci, ymin, ymax, color=color_markers)
plt.text(np.sqrt(wpi_sqr) / wci, y_text, '$\\omega_{pi}$')
plt.vlines(np.sqrt(wpe_sqr) / wci, ymin, ymax, color=color_markers)
plt.text(np.sqrt(wpe_sqr) / wci, y_text, '$\\omega_{pe}$')

for key in data_dict:
    print(key)
    y = copy.deepcopy(data_dict[key])
    y[y < 0] = np.nan
    plt.plot(omega_rf / wci, y, label='$' + key + '$', color=color_dict[key], linestyle='-', linewidth=2)
    y = copy.deepcopy(data_dict[key])
    y[y > 0] = np.nan
    plt.plot(omega_rf / wci, -y, color=color_dict[key], linestyle='--', linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\omega/ \\omega_{ci}$')
# plt.ylim([ymin, ymax])
plt.ylim([1e-5, 1e10])
plt.legend(loc=3)
plt.grid(True)
plt.tight_layout()
