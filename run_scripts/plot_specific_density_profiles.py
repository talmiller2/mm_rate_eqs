import matplotlib.pyplot as plt
import matplotlib.cm as cm

# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 14})

import numpy as np
from scipy.optimize import curve_fit

from mm_rate_eqs.relaxation_algorithm_functions import load_simulation


def define_plasma_mode_label(plasma_mode):
    label = ''
    if plasma_mode == 'isoT':
        label += 'isothermal'
    elif plasma_mode == 'isoTmfp':
        # label += 'isothermal iso-mfp'
        label += 'diffusion'
    elif 'cool' in plasma_mode:
        plasma_dimension = int(plasma_mode.split('d')[-1])
        label += 'cooling d=' + str(plasma_dimension)
    return label


def define_LC_mode_label(LC_mode):
    label = ''
    if LC_mode == 'sLC':
        label += 'with static-LC'
    else:
        label += 'with dynamic-LC'
    return label


def define_label(plasma_mode, LC_mode):
    label = define_plasma_mode_label(plasma_mode)
    label += ', '
    label += define_LC_mode_label(LC_mode)
    return label


plt.close('all')

# main_dir = '../runs/slurm_runs/set2_Rm_3/'
# main_dir = '../runs/slurm_runs/set4_Rm_3_mfp_over_cell_4/'
# main_dir = '../runs/slurm_runs/set5_Rm_3_mfp_over_cell_20/'
main_dir = '../runs/slurm_runs/set6_Rm_3_mfp_over_cell_1_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set7_Rm_3_mfp_over_cell_20_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set10_Rm_3_mfp_over_cell_0.04_mfp_limitX100/'
# main_dir = '../runs/slurm_runs/set11_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2/'
# main_dir = '../runs/slurm_runs/set12_Rm_3_mfp_over_cell_1_mfp_limitX100_nend_1e-2_rbc_adjut_ntL_timestepdef_without_ntL/'

plasma_modes = []
plasma_modes += ['isoTmfp']
plasma_modes += ['isoT']
# plasma_modes += ['coold1']
plasma_modes += ['coold2']
# plasma_modes += ['coold3']
# plasma_modes += ['cool']
# plasma_modes += ['cool_mfpcutoff']

# colors = cm.rainbow(np.linspace(0, 1, len(plasma_modes)))
colors = ['b', 'g', 'r', 'm', 'c', 'k']

# number_of_cells = 30
number_of_cells = 100
U = 0

linestyles = []
linestyles += ['-']
linestyles += ['--']

for ind_mode in range(len(plasma_modes)):
    plasma_mode = plasma_modes[ind_mode]
    color = colors[ind_mode]

    run_name = plasma_mode
    run_name += '_N_' + str(number_of_cells) + '_U_' + str(U)
    run_name += '_sLC'
    # print('run_name = ' + run_name)

    save_dir = main_dir + '/' + run_name

    state_file = save_dir + '/state.pickle'
    settings_file = save_dir + '/settings.pickle'
    try:
        state, settings = load_simulation(state_file, settings_file)

        label = define_plasma_mode_label(plasma_mode)
        plt.figure(1)
        x = np.linspace(0, number_of_cells, number_of_cells)
        plt.plot(x, state['n_tR'], label='$n_{tR}$ ' + label, linestyle='solid', color=color)
        plt.plot(x, state['n_tL'], label='$n_{tL}$ ' + label, linestyle='dashed', color=color)
        plt.plot(x, state['n_c'], label='$n_{c}$ ' + label, linestyle='dashdot', color=color)
        plt.xlabel('cell number')
        plt.ylabel('[$m^{-3}$]')
        plt.title('density profiles for N=' + str(number_of_cells))
        plt.tight_layout()
        plt.grid(True)
        plt.legend()

        plt.figure(2)
        x = np.linspace(0, number_of_cells, number_of_cells)
        plt.plot(x, state['mean_free_path'], label=label, linestyle='solid', color=color)
        plt.yscale('log')
        plt.xlabel('cell number')
        plt.ylabel('[$m$]')
        plt.title('mean free path profiles for N=' + str(number_of_cells))
        plt.tight_layout()
        plt.grid(True)
        plt.legend()

    except:
        pass
