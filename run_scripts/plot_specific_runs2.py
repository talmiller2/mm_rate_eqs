import matplotlib.pyplot as plt
import numpy as np

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation, plot_relaxation_status

plt.close('all')

save_dirs = []
linestyles = []

# linestyles = ['-', '--']
linestyles = ['-', '--', ':', '-.']

main_dir = '../runs/slurm_runs/set2_Rm_3/'

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
plasma_modes += ['cool']
plasma_modes += ['cool_mfpcutoff']
# plasma_modes += ['cool_d1']

curr_LC_mode = 'sLC'
LC_modes = [curr_LC_mode for i in range(len(plasma_modes))]

curr_num_cells = 100
num_cells_list = [curr_num_cells for _ in range(len(plasma_modes))]

curr_U = 0.5
U_list = [curr_U for _ in range(len(plasma_modes))]

for i, linestyle in enumerate(linestyles):
    run_name = plasma_modes[i]
    run_name += '_N_' + str(num_cells_list[i]) + '_U_' + str(U_list[i])
    run_name += '_' + LC_modes[i]
    print('run_name = ' + run_name)

    save_dir = main_dir + '/' + run_name
    # print(save_dir)
    state_file = save_dir + '/state.pickle'
    settings_file = save_dir + '/settings.pickle'
    state, settings = load_simulation(state_file, settings_file)

    print('flux = ' + str(state['flux_mean']))

    if state['successful_termination'] == False:
        print('RUN FAILED.')

    settings['linestyle'] = linestyle
    plot_relaxation_status(state, settings)
