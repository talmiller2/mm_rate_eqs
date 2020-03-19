import copy

import matplotlib.pyplot as plt
import numpy as np


def getSolidAngles(U, vth, alpha):
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


plt.close('all')

# Solve analytic zeros of loss-cone relation
vth = 1.0
# vth = 2.0
# vth = 734263.3729228973
# vth = 728838.6716588414

# Rm_list = np.array([1.4])
# Rm_list = np.array([2.0])
# colors = ['k']
# Rm_list = np.array([1.4, 1.6, 1.8, 2.0])
# Rm_list = np.array([1.4, 2.0, 2.5, 3.0])
# colors = ['k', 'b', 'r', 'g']
Rm_list = np.array([1.4, 1.7, 2.0])
colors = ['b', 'orange', 'g']
for ind_Rm, Rm in enumerate(Rm_list):
    alpha = 1 / Rm
    U_list = np.linspace(0, 2.0 * vth, 1000)
    #    U_list = np.linspace(0, 0.3*vth, 1000)
    v_perp_high = np.nan * U_list
    v_perp_low = np.nan * U_list

    omega_tR = np.zeros(len(U_list))
    omega_tL = np.zeros(len(U_list))
    omega_c = np.zeros(len(U_list))

    U_transition = np.sqrt((1 - alpha) / alpha * vth ** 2)
    U_last_sol = np.sqrt(vth ** 2 / alpha)
    v_perp_last_sol = np.sqrt((1 - alpha) * vth ** 2)

    for ind_U, U in enumerate(U_list):

        #        a = 1.0
        #        b = 2*alpha*( U**2*(2*alpha - 1) - vth**2)
        #        c = alpha**2*( vth**2 - U**2 )**2
        #        det = b**2 - 4*a*c
        #
        #        v_perp_squared_sol_high = np.nan
        #        v_perp_squared_sol_low = np.nan
        #        if det>=0:
        #            v_perp_squared_sol_high = (-b + np.sqrt(det))/(2*a)
        #            v_perp_squared_sol_low = (-b - np.sqrt(det))/(2*a)
        ##        else:
        ##            v_perp_squared_sol_high = -b/(2*a)
        ##            v_perp_squared_sol_low = -b/(2*a)
        #        if v_perp_squared_sol_high>vth**2: v_perp_squared_sol_high = vth**2
        #        if v_perp_squared_sol_low<0: v_perp_squared_sol_low = 0
        #        v_perp_high[ind_U] = np.sqrt(v_perp_squared_sol_high)
        #        v_perp_low[ind_U] = np.sqrt(v_perp_squared_sol_low)

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
        v_perp_high[ind_U] = np.sqrt(v_perp_squared_norm_sol_high) * vth
        v_perp_low[ind_U] = np.sqrt(v_perp_squared_norm_sol_low) * vth

        do_test = False
        if do_test:
            # check place in the inequality
            print('v_perp_squared_sol_high=', v_perp_squared_sol_high)
            print('v_perp_squared_sol_low=', v_perp_squared_sol_low)

            v_perp_squared_list = [(v_perp_squared_sol_high + vth ** 2) / 2,
                                   (v_perp_squared_sol_high + v_perp_squared_sol_low) / 2,
                                   (v_perp_squared_sol_low) / 2]
            for v_perp_squared in v_perp_squared_list:
                print('v_perp_squared=', v_perp_squared)
                ineq = v_perp_squared / (v_perp_squared + (np.sqrt(vth ** 2 - v_perp_squared) - U) ** 2) - alpha
                print('left particles ineq=', ineq)
                if ineq < 0:
                    print('in LC')
                else:
                    print('not in LC')
                ineq = v_perp_squared / (v_perp_squared + (-np.sqrt(vth ** 2 - v_perp_squared) - U) ** 2) - alpha
                print('right particles ineq=', ineq)
                if ineq < 0:
                    print('in LC')
                else:
                    print('not in LC')

        # calculate the solid angles for right/left loss cones
        omega_tR[ind_U], omega_tL[ind_U], omega_c[ind_U] = getSolidAngles(U, vth, alpha)

    plt.figure(2)
    plt.plot(U_list / vth, v_perp_high / vth, label='high Rm=' + str(Rm), color=colors[ind_Rm])
    plt.plot(U_list / vth, v_perp_low / vth, '--', label='low Rm=' + str(Rm), color=colors[ind_Rm])
    plt.plot(U_transition / vth, vth / vth, 'o', color=colors[ind_Rm])
    plt.plot(U_last_sol / vth, v_perp_last_sol / vth, 'o', color=colors[ind_Rm])

    # plot area of right-LC
    v_right_LC_max = copy.deepcopy(v_perp_high)
    for i, U in enumerate(U_list):
        if U > U_transition: v_right_LC_max[i] = vth
    plt.figure(3)
    plt.fill_between(U_list / vth, v_right_LC_max / vth, color=colors[ind_Rm], alpha=0.5, label='Rm=' + str(Rm),
                     linewidth=0.0)

    # plot area of left-LC
    v_left_LC_low = copy.deepcopy(v_perp_low)
    ind_transition = np.where(U_list > U_transition)[0][0]
    inds_transition = range(ind_transition, len(U_list))
    v_left_LC_high = copy.deepcopy(v_perp_high)
    ind_last_sol = np.where(U_list > U_last_sol)[0][0]
    inds_last_sol = range(ind_last_sol, len(U_list))
    plt.figure(4)
    plt.fill_between(U_list / vth, v_left_LC_low / vth, color=colors[ind_Rm], alpha=0.5, label='Rm=' + str(Rm),
                     linewidth=0.0)
    plt.fill_between(U_list[inds_transition] / vth, y1=v_left_LC_high[inds_transition] / vth, y2=vth / vth,
                     color=colors[ind_Rm], alpha=0.5, linewidth=0.0)
    plt.fill_between(U_list[inds_last_sol] / vth, vth / vth, color=colors[ind_Rm], alpha=0.5, linewidth=0.0)

    # calculate the solid angles for right/left loss cones
    theta_low = np.arcsin(v_perp_low / vth)
    theta_high = np.arcsin(v_perp_high / vth)
    plt.figure(5)
    plt.plot(U_list / vth, theta_low / np.pi, label='high Rm=' + str(Rm), color=colors[ind_Rm])
    plt.plot(U_list / vth, theta_high / np.pi, '--', label='low high Rm=' + str(Rm), color=colors[ind_Rm])

    #    omega_tR = np.sin(theta_high/2)**2
    #    omega_tR[inds_transition] = 0.5
    #    omega_tL = np.sin(theta_low/2)**2
    #    omega_tL[inds_transition] = np.sin(theta_low[inds_transition]/2)**2 + 0.5 - np.sin(theta_high[inds_transition]/2)**2
    #    omega_tL[inds_last_sol] = 0.5
    #    omega_c = 1 - omega_tR - omega_tL

    plt.figure(6)
    plt.plot(U_list / vth, omega_tR, label='$\\Omega_{tR}$ Rm=' + str(Rm), color=colors[ind_Rm])
    plt.plot(U_list / vth, omega_tL, '--', label='$\\Omega_{tL}$ Rm=' + str(Rm), color=colors[ind_Rm])
    plt.plot(U_list / vth, omega_c, ':', label='$\\Omega_{c}$ Rm=' + str(Rm), color=colors[ind_Rm])

    # ratio that should hold at steady state
    #    vth_reduced = 0.3*vth
    #    flux = (vth_reduced - U_list*omega_c/omega_tR)/vth
    #    flux = (vth - U_list*omega_c/omega_tR)/vth
    flux = (vth * (1 - omega_tL / omega_tR) - U_list * omega_c / omega_tR) / vth
    plt.figure(7)
    plt.plot(U_list / vth, flux, label='Rm=' + str(Rm), color=colors[ind_Rm])
#    plt.plot(U_list/vth, omega_c/omega_tR, label='Rm='+str(Rm), color=colors[ind_Rm])          

plt.figure(2)
plt.xlabel('U/vth')
plt.ylabel('v_perp/vth')
plt.title('Critical v_perp values')
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure(3)
plt.xlabel('U/vth')
plt.ylabel('v_perp/vth')
plt.title('Right loss-cone v_perp range')
plt.legend()
plt.tight_layout()

plt.figure(4)
plt.xlabel('U/vth')
plt.ylabel('v_perp/vth')
plt.title('Left loss-cone v_perp range')
plt.legend()
plt.tight_layout()

plt.figure(5)
plt.xlabel('U/vth')
plt.ylabel('$\\theta$/$\\pi$')
plt.title('Critical angles')
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure(6)
plt.xlabel('U/vth')
plt.ylabel('$\\Omega$ /4$\\pi$')
plt.title('Solid angles')
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure(7)
plt.xlabel('U/vth')
plt.title('Flux factor')
plt.legend()
plt.grid()
plt.tight_layout()
