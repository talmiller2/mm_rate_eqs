import numpy as np


def calc_axial_velocity_loss_cone_solutions(U, v, Rm):
    # solve analytical values of critical v_z
    vz_sol1 = -1 / Rm * U + 1 / Rm * np.sqrt((Rm - 1) * (Rm * v ** 2 - U ** 2))
    vz_sol2 = -1 / Rm * U - 1 / Rm * np.sqrt((Rm - 1) * (Rm * v ** 2 - U ** 2))
    return vz_sol1, vz_sol2


def calc_special_U_values(v, Rm):
    U_transition = np.sqrt((Rm - 1) * v ** 2)
    U_last_sol = np.sqrt(Rm * v ** 2)
    return U_transition, U_last_sol


def calc_modified_loss_cone_angle(vz, v, U):
    vr = np.sqrt(v ** 2 - vz ** 2)
    modified_theta = np.arcsin(vr / v)
    return modified_theta


def get_solid_angles(U, v, alpha):
    Rm = 1 / alpha  # keep alpha to satisfy previous argument structure
    vz_sol1, vz_sol2 = calc_axial_velocity_loss_cone_solutions(U, v, Rm)
    U_transition, U_last_sol = calc_special_U_values(v, Rm)
    theta_high = calc_modified_loss_cone_angle(vz_sol1, v, U)
    theta_low = calc_modified_loss_cone_angle(vz_sol2, v, U)

    ## solid angles of different species ###

    # omega_tL is defined only by its original LC that shrinks with U till U=v where it vanishes.
    if U <= v:
        omega_tL = np.sin(theta_low / 2) ** 2
    else:
        omega_tL = 0

    # omega_tR has the original right cone below U=v, and above also the original left cone.
    # above U_transition the right-cone becomes inverted, and above U_last_sol it consumes all angles.
    if U <= v:
        omega_tR = np.sin(theta_high / 2) ** 2
    elif U > v and U <= U_transition:
        omega_tR = np.sin(theta_high / 2) ** 2 + np.sin(theta_low / 2) ** 2
    elif U > U_transition and U <= U_last_sol:
        omega_tR = 1 - np.sin(theta_high / 2) ** 2 + np.sin(theta_low / 2) ** 2
    else:
        omega_tR = 1

    # omega_c will be defined as the remainder of the total solid angle
    omega_c = 1 - omega_tR - omega_tL

    return omega_tR, omega_tL, omega_c
