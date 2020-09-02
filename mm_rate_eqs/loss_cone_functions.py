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

    # v, theta at the critical value
    v_perp_last_sol = np.sqrt((1 - alpha) * vth ** 2)
    theta_last_sol = np.arcsin(v_perp_last_sol / vth)

    ## solid angles of different species ###

    # omega_tL is defined only by its original LC that shrinks with U till v_th where it vanishes
    if U <= vth:
        omega_tL = np.sin(theta_low / 2) ** 2
    else:
        omega_tL = 0

    # omega_tR has 3 contributions:
    # 1) the original right LC that grows with U till U_transition where it is saturated
    # 2) the original left LC that starts contributing above v_th till U_last_sol where it saturates
    # 3) the phase space that was originally outside of the left LC and enters the right LC,
    #    contributes between U_transition and U_last_sol

    # contribution 1
    if U <= U_transition:
        omega_tR_1 = np.sin(theta_high / 2) ** 2
    else:
        omega_tR_1 = 0.5

    # contribution 2
    if U <= vth:
        omega_tR_2 = 0
    elif U > vth and U <= U_last_sol:
        omega_tR_2 = np.sin(theta_low / 2) ** 2
    else:
        omega_tR_2 = np.sin(theta_last_sol / 2) ** 2

    # contribution 3
    if U <= U_transition:
        omega_tR_3 = 0
    elif U > U_transition and U <= U_last_sol:
        omega_tR_3 = 0.5 - np.sin(theta_high / 2) ** 2
    else:
        omega_tR_3 = 0.5 - np.sin(theta_last_sol / 2) ** 2

    # combine contributions
    omega_tR = omega_tR_1 + omega_tR_2 + omega_tR_3

    # omega_c will be defined as the remainder of the total solid angle
    omega_c = 1 - omega_tR - omega_tL

    return omega_tR, omega_tL, omega_c
