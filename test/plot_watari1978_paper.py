import copy
import numpy as np
from mm_rate_eqs.constants_functions import define_electron_mass, define_proton_mass, define_factor_eV_to_K, \
    define_boltzmann_constant, define_factor_Pa_to_bar, define_vacuum_permeability, define_electron_charge, \
    define_vacuum_permittivity, define_speed_of_light
from mm_rate_eqs.plasma_functions import get_debye_length
import matplotlib.colors as colors

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
plt.close('all')


def plot_pos_neg(x, y, color=None, label=None):
    y_copy = copy.deepcopy(y)
    y_copy[y < 0] = np.nan
    plt.plot(x, y_copy, label=label, color=color, linestyle='-', linewidth=2)
    y_copy = copy.deepcopy(y)
    y_copy[y > 0] = np.nan
    plt.plot(x, -y_copy, color=color, linestyle='--', linewidth=2)


def num_to_coords_str(x, notation_type='both', scilimits=(-1e5, 1e6)):
    if notation_type in ['f', 'float']:
        format_type = 'f'
    elif notation_type in ['e', 'scientific']:
        format_type = 'e'
    else:
        if x < scilimits[0] or x > scilimits[1]:
            format_type = 'e'
        else:
            format_type = 'f'
    return ('{:.4' + format_type + '}').format(x)


def format_coord(x, y, X, Y, Z, **notation_kwargs):
    coords_str = 'x=' + num_to_coords_str(x, **notation_kwargs) + ', y=' + num_to_coords_str(y, **notation_kwargs)
    xarr = X[0, :]
    yarr = Y[:, 0]
    if ((x > xarr.min()) & (x <= xarr.max()) &
            (y > yarr.min()) & (y <= yarr.max())):
        col = np.searchsorted(xarr, x) - 1
        row = np.searchsorted(yarr, y) - 1
        z = Z[row, col]
        coords_str += ', z=' + num_to_coords_str(z, **notation_kwargs) + '  [' + str(row) + ',' + str(col) + ']'
    return coords_str


def update_format_coord(X, Y, Z, ax=None, **notation_kwargs):
    if ax == None:
        ax = plt.gca()
    ax.format_coord = lambda x, y: format_coord(x, y, X, Y, Z, **notation_kwargs)
    plt.show()


e = define_electron_charge()
me = define_electron_mass()
mp = define_proton_mass()
eps0 = define_vacuum_permittivity()
kB = define_boltzmann_constant()
c = define_speed_of_light()

# fusion like parameters
Z = 1
A = 2  # deuterium
B0 = 1  # [T]
n = 1e21  # [m^-3]
# n = 1e19 # [m^-3]
ni = n  # [m^-3]
ne = n  # [m^-3]
Te_keV = 10
# Te_keV = 0.1
Te_eV = Te_keV * 1e3
Ti_eV = Te_eV
# l = 0.1 # [m]
# l = 1  # [m]
# l = 10 # [m]
l = 100  # [m]

# # Watari paper parameters
# Z = 2
# A = 4
# B0_gauss = 12.2e3 # [G]
# B0 = B0_gauss / 1e4 # [T]
# ni = 1.5e19 # [m^-3]
# ne = 1.5e19 # [m^-3]
# Te_eV = 13
# Ti_eV = 15
# lamda = 0.005 # plasma width [m]
# l = 1 # [m]


Te_K = Te_eV * define_factor_eV_to_K()
Ti_K = Ti_eV * define_factor_eV_to_K()
mi = A * mp
qi = Z * e
omega_ce = e * B0 / me
omega_ci = Z * e * B0 / mi
v_th_i = np.sqrt(kB * Ti_K / mi)
r_ci = v_th_i / omega_ci
omega_pe = np.sqrt(ne * e ** 2 / (eps0 * me))
omega_pi = np.sqrt(ni * qi ** 2 / (eps0 * mi))
lambda_debye = np.sqrt(eps0 * kB * Te_K / (ne * e ** 2))
k_debye = 1 / lambda_debye
gamma_i = 3.0
cs = np.sqrt(gamma_i * kB * Ti_K / mi)
kz = np.pi / l

lamda = 0.5  # plasma width [m]
# lamda = 1e-3 # plasma width [m]

# # test against Watari paper (see Fig. 12)
# print('*** Compare numbers to written in the Watari 1978 paper Fig 12:')
# print('(omega_pi / omega_ci) ^ 2 = ', (omega_pi / omega_ci) ** 2)
# print('paper value = ', 7600)
# print('Ti_K * lamda ^ 2 / (Te_K * r_ci ^ 2) = ', Ti_K * lamda ** 2 / (Te_K * r_ci ** 2))
# print('paper value = ', 275)
# print('(c / omega_pi / lamda) ^ 2 = ', (c / omega_pi / lamda) ** 2)
# print('paper value = ', 139)


def get_eps(omega, lamda, omega_pi):
    eps = 1 - omega_pi ** 2 / (omega ** 2 - omega_ci ** 2) \
          + k_debye ** 2 * lamda ** 2 * (1 - (cs * kz / omega) ** 2) / (1 - (omega * k_debye * lamda / (c * kz)) ** 2)
    return eps


def get_eps_tag(omega, lamda, omega_pi):
    eps_tag = 1 - omega_pi ** 2 \
              / (omega ** 2 - omega_ci ** 2
                 + (omega_pi * omega * lamda / c) ** 2 * (1 - ((omega_pi * omega) / (c * kz * omega_ci)) ** 2)) \
              + 0.5 * (k_debye * lamda) ** 2 / (1 - 0.5 * (omega * k_debye * lamda / (c * kz)) ** 2)
    return eps_tag


def get_fields_for_cases_a_c(omega, lamda, omega_pi, Ex0, By0):
    """
    formula for cases (a) and (c),
    where (c) analogous to type III (external By0), and (a) is capacitor-like (external Ex0)
    """
    eps = get_eps(omega, lamda, omega_pi)
    Ex = (Ex0 + k_debye ** 2 * lamda ** 2 * omega * By0 / (c * kz) / (
                1 - (omega * k_debye * lamda / (c * kz)) ** 2)) / eps
    Ey = omega / omega_ci * (omega_pi * lamda * omega / c) ** 2 \
         / (omega ** 2 - omega_ci ** 2 + (omega_pi * lamda * omega / c) ** 2) * Ex  # without the factor of i
    By = By0 / (1 - (omega_pi * omega / (c * kz)) ** 2 / (cs ** 2 / lamda ** 2 - omega ** 2 + omega_ci ** 2))
    return Ex, Ey, By
    # return abs(Ex), abs(Ey), abs(By)


def get_fields_for_case_a(omega, lamda, omega_pi, Ex0=1, By0=0):
    return get_fields_for_cases_a_c(omega, lamda, omega_pi, Ex0=Ex0, By0=By0)


def get_fields_for_case_c(omega, lamda, omega_pi, Ex0=0, By0=1):
    return get_fields_for_cases_a_c(omega, lamda, omega_pi, Ex0=Ex0, By0=By0)


def get_field_case_b(omega, lamda, omega_pi, Bz0=1):
    """
    formula for case (b), which is analogous to types I/II (external Bz0 with regular coil)
    """
    eps_tag = get_eps_tag(omega, lamda, omega_pi)
    dEx_dx = - omega ** 2 * Bz0 ** 2 / (c * omega_ci * eps_tag) \
             / ((omega ** 2 - omega_ci ** 2) / omega_pi ** 2
                + (omega * lamda / c) ** 2 * (1 - (omega * omega_pi / (c * kz * omega_ci)) ** 2))
    eps_mod = get_eps(omega, lamda / np.sqrt(2), omega_pi)
    Ez = - kz / k_debye ** 2 * dEx_dx \
         * (omega_pi ** 2 / (
                omega ** 2 - omega_ci ** 2) + omega_ci ** 2 / omega ** 2 * eps_mod)  # without the factor of i
    return dEx_dx, Ez


## plot 1d section
# omega = 0.1 * omega_ci
# omega = 0.5 * omega_ci
# omega = 0.95 * omega_ci
omega = 1.1 * omega_ci
# omega = 1e3 * omega_ci
# lamda = np.linspace(1e-10, 0.1, 300)
# lamda = np.logspace(-10, -1, 100)
lamda = np.logspace(-3, 1, 100)
# lamda = np.linspace(0.1, 2, 100)
plt.figure(3)
# Ex, Ey, By = get_fields_for_case_a(omega, lamda, omega_pi)
# plot_pos_neg(lamda, Ex, color='r', label='Ex (capacitor-like)')
# plot_pos_neg(lamda, Ey, color='g', label='Ey (capacitor-like)')
# plot_pos_neg(lamda, By, color='m', label='By (capacitor-like)')
Ex, Ey, By = get_fields_for_case_c(omega, lamda, omega_pi)
plot_pos_neg(lamda, Ex, color='r', label='Ex (type III)')
plot_pos_neg(lamda, Ey, color='g', label='Ey (type III)')
plot_pos_neg(lamda, By, color='m', label='By (type III)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\lambda$ [m]')
plt.ylabel('field amplitude')
plt.legend()
plt.tight_layout()

## plot 2d ralations as a function of n, omega
# lamda = 0.001  # [m]
lamda = 0.1  # [m]
# lamda = 0.5 # [m]
omega = np.logspace(-2, 2, 300) * omega_ci  # [Hz]
ni = np.logspace(15, 21, 300)  # [m^-3]
omega_2d, ni_2d = np.meshgrid(omega, ni)
omega_pi_2d = np.sqrt(ni_2d * qi ** 2 / (eps0 * mi))
# ni_2d /= 1e15  # just for plot to test
cmap_pos = 'hot_r'
cmap_neg = 'Greys'

## case (c) - type III
vmin, vmax = 1e-2, 1e2
# vmin, vmax = 1e-3, 1e3
Ex, Ey, By = get_fields_for_case_c(omega_2d, lamda, omega_pi_2d)
X, Y, Z = omega_2d / omega_ci, ni_2d, By
plt.figure(4, figsize=(7, 5))
# plt.figure(4, figsize=(12, 5))
# plt.subplot(1, 3, 1)
plt.pcolormesh(X, Y, Z,
               norm='log',
               vmin=vmin,
               vmax=vmax,
               cmap=cmap_pos,
               )
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.pcolormesh(X, Y, -Z,
               norm='log',
               vmin=vmin,
               vmax=vmax,
               cmap=cmap_neg,
               )
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\omega / \\omega_{ci}$')
plt.ylabel('$n \; [m^{-3}]$')
plt.title('case (c) $\\frac{B_y}{B_{y0}}$ for $T$=' + str(Te_keV) + 'keV, ' + '$\\frac{k_z}{\\pi}$=' + str(
    kz / np.pi) + '$m^{-1}$, $\\lambda$=' + str(lamda) + 'm', pad=20)
plt.colorbar()
plt.tight_layout()
update_format_coord(X, Y, Z)

# case (a) - capacitor-like
Ex, Ey, By = get_fields_for_case_a(omega_2d, lamda, omega_pi_2d)
X, Y, Z = omega_2d / omega_ci, ni_2d, Ex
plt.figure(5, figsize=(7, 5))
# plt.subplot(1, 3, 2)
plt.pcolormesh(X, Y, Z,
               norm='log',
               vmin=vmin,
               vmax=vmax,
               cmap=cmap_pos,
               )
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.pcolormesh(X, Y, -Z,
               norm='log',
               vmin=vmin,
               vmax=vmax,
               cmap=cmap_neg,
               )
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\omega / \\omega_{ci}$')
plt.ylabel('$n \; [m^{-3}]$')
plt.title('case (a) $\\frac{E_x}{E_{x0}}$ for $T$=' + str(Te_keV) + 'keV, ' + '$\\frac{k_z}{\\pi}$=' + str(
    kz / np.pi) + '$m^{-1}$, $\\lambda$=' + str(lamda) + 'm', pad=20)
plt.colorbar()
plt.tight_layout()
update_format_coord(X, Y, Z)

# case (b) - types I-II
dEx_dx, Ez = get_field_case_b(omega_2d, lamda, omega_pi_2d)
X, Y, Z = omega_2d / omega_ci, ni_2d, dEx_dx
plt.figure(6, figsize=(7.5, 5.5))
# plt.subplot(1, 3, 3)
plt.pcolormesh(X, Y, Z,
               norm='log',
               vmin=vmin,
               vmax=vmax,
               cmap=cmap_pos,
               )
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.pcolormesh(X, Y, -Z,
               norm='log',
               vmin=vmin,
               vmax=vmax,
               cmap=cmap_neg,
               )
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\omega / \\omega_{ci}$')
plt.ylabel('$n \; [m^{-3}]$')
plt.title('case (b) $\\frac{\\partial E_x}{\\partial x} \\frac{V}{m^2}$ for $B_{z0}=1$T, T=' + str(
    Te_keV) + 'keV, ' + '$\\frac{k_z}{\\pi}$=' + str(kz / np.pi) + '$m^{-1}$, $\\lambda$=' + str(lamda) + 'm', pad=20)
plt.colorbar()
plt.tight_layout()
update_format_coord(X, Y, Z)

