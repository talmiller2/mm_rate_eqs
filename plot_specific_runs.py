import matplotlib.pyplot as plt

from relaxation_algorithm_functions import load_simulation, plot_relaxation_status

plt.close('all')

save_dirs = []
linestyles = []

# linestyles = ['-', '--']
linestyles = ['-', '--', ':']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_iso_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_iso_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_iso_rbc_none_energy_scheme_none']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_cool_d_3_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_cool_d_3_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_cool_d_3_rbc_none_energy_scheme_none']

save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_cool_d_1_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.05_trans_none_cool_d_1_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_cool_d_1_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.2_trans_none_cool_d_1_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_cool_d_1_rbc_none_energy_scheme_none']

for save_dir, linestyle in zip(save_dirs, linestyles):
    print(save_dir)
    state_file = save_dir + '/state.pickle'
    settings_file = save_dir + '/settings.pickle'
    state, settings = load_simulation(state_file, settings_file)
    # settings['save_plots'] = False # or it will override existing plots
    settings['linestyle'] = linestyle
    plot_relaxation_status(state, settings)
