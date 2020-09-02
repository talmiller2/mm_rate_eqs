import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation, plot_relaxation_status

plt.close('all')

save_dirs = []
linestyles = []

# linestyles = ['-', '--']
linestyles = ['-', '--', ':']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_iso_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_iso_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_iso_rbc_none_energy_scheme_none']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.05_trans_none_iso_rbc_none_energy_scheme_none_constLC']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_iso_rbc_none_energy_scheme_none_constLC']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_iso_rbc_none_energy_scheme_none_constLC']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.5_trans_none_iso_rbc_none_energy_scheme_none_constLC']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_iso_rbc_none_energy_scheme_none_constLC_const_dens']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_iso_rbc_none_energy_scheme_none_constLC_const_dens']
save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_iso_rbc_none_energy_scheme_none_constLC_const_dens']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_iso_rbc_none_energy_scheme_none_const_dens']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_iso_rbc_none_energy_scheme_none_const_dens']
save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_iso_rbc_none_energy_scheme_none_const_dens']

save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_iso_rbc_none_energy_scheme_none_Ufac0.5_const_dens']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_cool_d_3_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_cool_d_3_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_cool_d_3_rbc_none_energy_scheme_none']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_cool_d_1_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.05_trans_none_cool_d_1_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.1_trans_none_cool_d_1_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.2_trans_none_cool_d_1_rbc_none_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.3_trans_none_cool_d_1_rbc_none_energy_scheme_none']

# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_cool_d_1_rbc_none_energy_scheme_none_dt_factor_3']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_cool_d_1_rbc_nullify_ntL_factor_0.01_energy_scheme_none']
# save_dirs += ['runs/runs_August_2020/test_N_40_U_0_trans_none_cool_d_1_rbc_none_energy_scheme_none']


# save_dirs += ['runs/runs_August_2020/test_N_30_U_0_trans_none_cool_d_1_rbc_none_energy_scheme_none_constLC']
# save_dirs += ['runs/runs_August_2020/test_N_30_U_0.05_trans_none_cool_d_1_rbc_none_energy_scheme_none_constLC']


for save_dir, linestyle in zip(save_dirs, linestyles):
    print(save_dir)
    state_file = save_dir + '/state.pickle'
    settings_file = save_dir + '/settings.pickle'
    state, settings = load_simulation(state_file, settings_file)
    # print('alpha_definition = ' + str(settings['alpha_definition']))
    print('flux = ' + str(np.nanmean(state['flux'])))
    settings['linestyle'] = linestyle
    plot_relaxation_status(state, settings)
