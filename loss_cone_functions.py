import numpy as np


def get_solid_angles(U, vth, alpha):
    # solve analytical values of critical v_perp
    a = 1.0
    b = 2 * alpha * ((U / vth) ** 2 * (2 * alpha - 1) - 1)
    c = alpha ** 2 * (1 - (U / vth) ** 2) ** 2
    det = b ** 2 - 4 * a * c

    v_perp_squared_norm_sol_high = np.nan
    v_perp_squared_norm_sol_low = np.nan
    if det >= 0:
        v_perp_squared_norm_sol_high = (-b + np.sqrt(det)) / (2 * a)
        v_perp_squared_norm_sol_low = (-b - np.sqrt(det)) / (2 * a)

    if v_perp_squared_norm_sol_high > 1: v_perp_squared_norm_sol_high = 1.0
    if v_perp_squared_norm_sol_low < 0: v_perp_squared_norm_sol_low = 0
    v_perp_high = np.sqrt(v_perp_squared_norm_sol_high) * vth
    v_perp_low = np.sqrt(v_perp_squared_norm_sol_low) * vth

    # translate to angles
    theta_low = np.arcsin(v_perp_low / vth)
    theta_high = np.arcsin(v_perp_high / vth)

    # critical U values for loss-cones
    U_transition = np.sqrt((1 - alpha) / alpha * vth ** 2)
    U_last_sol = np.sqrt(vth ** 2 / alpha)

    # translate to solid angles of loss-cones
    if U <= U_transition:
        omega_tR = np.sin(theta_high / 2) ** 2
    else:
        omega_tR = 0.5
    if U <= U_transition:
        omega_tL = np.sin(theta_low / 2) ** 2
    elif U > U_transition and U <= U_last_sol:
        omega_tL = np.sin(theta_low / 2) ** 2 + 0.5 - np.sin(theta_high / 2) ** 2
    else:
        omega_tL = 0.5
    omega_c = 1 - omega_tR - omega_tL

    return omega_tR, omega_tL, omega_c
