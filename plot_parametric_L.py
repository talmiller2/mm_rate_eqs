import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})

from default_settings import define_default_settings
from relaxation_algorithm_functions import load_simulation

# plt.close('all')

# parametric scan
# save_dir_main = 'runs/runs_April_2020/runs_no_transition_different_number_of_cells/'
# number_of_cells_list = np.round(np.linspace(5,150,15))[0:-1]

# nullify_ntL_factor = 0.05
# # nullify_ntL_factor = 0.01
# label = 'isothermal_nullify_ntL_factor_' + str(nullify_ntL_factor)
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_nullify_ntL_factor_' \
#                 + str(nullify_ntL_factor)
# label = 'isothermal_rbc_none_energycons_none_U_0'
# label = 'isothermal U=0'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none'
# label = 'isothermal_rbc_none_energycons_none_U_0.3'
# save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none_U_0.3'

mode = 'iso'
# mode = 'cool'

# d = 3
d = 1

# U = 0
# linestyle = '-'
U = 0.3
linestyle = '--'

label = mode
save_dir_main = 'runs/runs_August_2020/different_number_of_cells_rbc_none_energycons_none'
save_dir_main += '_' + mode
if mode == 'cool':
    label += ' d=' + str(d)
    save_dir_main += '_d_' + str(d)
label += ', U=' + str(U)
save_dir_main += '_U_' + str(U)
# save_dir_main += '_adaptive_mirror_mfp'


number_of_cells_list = np.round(np.linspace(5, 100, 15))

flux_list = np.nan * np.zeros(len(number_of_cells_list))

for i, number_of_cells in enumerate(number_of_cells_list):
    # print('number_of_cells=' + str(int(number_of_cells)))

    save_dir = save_dir_main + '/number_of_cells_' + str(int(number_of_cells))

    try:  # will fail if run does not exist
        state_file = save_dir + '/state.pickle'
        settings_file = save_dir + '/settings.pickle'
        state, settings = load_simulation(state_file, settings_file)
        flux_list[i] = state['flux_mean']
    except:
        pass

    # extract the density profile
    chosen_num_cells = 32
    if number_of_cells == chosen_num_cells:
        plt.figure(2)
        plt.plot(state['n'], '-', label=label, linestyle=linestyle)

plt.figure(1)
plt.plot(number_of_cells_list, flux_list, '-', label=label, linestyle=linestyle)

# p = np.polyfit(number_of_cells_list, flux_list, deg=2)
# flux_poly2_fit = np.polyval(p, number_of_cells_list)
# plt.plot(number_of_cells_list, flux_poly2_fit, '-', label='quad fit')

plt.figure(1)
plt.xlabel('number of cells')
plt.ylabel('flux')
plt.title('flux as a function of cells in the system')
plt.tight_layout()
plt.grid(True)
plt.legend()

plt.figure(2)
plt.xlabel('cell number')
plt.ylabel('density')
plt.title('density profile for N = ' + str(chosen_num_cells) + ' cells')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.grid(True)
