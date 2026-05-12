import numpy as np
from scipy.stats import qmc
#from expandLHS import ExpandLHS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch.nn as nn
import torch.nn.functional as F

import py21cmfast as p21c
from py21cmemu.properties import emulator_properties
import tools21cm

from types import SimpleNamespace
import h5py


def lhs_sampler(num_rounds, label):

    #test_param = [-1.3, 0.5, -1.0, -0.5, 8.7, 0.5, 40.5, 500.0, 1.0]
    # column = ['F_STAR10', 'ALPHA_STAR', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 't_STAR', 'L_X', 'NU_X_THRESH', 'X_RAY_SPEC_INDEX']

        

    # lower_boundaries = [-3.0, -0.5, -3.0, -1.0, 8.0, 0.1, 38.0, 100.0, -1.0]
    # upper_boundaries = [-0.05, 1.0, -0.05, 0.5, 10.0, 1.0, 42.0, 1500.0, 3.0]

    # lower_boundaries = [-3.0, -0.5, -3.0, -1.0, 8.0]
    # upper_boundaries = [-0.05, 1.0, -0.05, 0.5, 10.0]



    if label == 'TrainingData':
        path = 'training_data_input_2986'
        #n_samples = 2986

    elif label == 'ValidationData':
        path = 'validation_data_input_746'
        #n_samples = 746

    else:
        path = 'test_data_input_933'
        #n_samples = 933

    round_points = []
    # starting_point = 1
    # sample = None

    for i in range(1, num_rounds + 1):
        if os.path.exists(f'GeneratedData/Input/{label}/{path}_r{i}.h5'):
            print('Loading')
            data_input = pd.read_hdf(f'GeneratedData/Input/{label}/{path}_r{i}.h5')
            round_points.append(data_input)
            #unscaled_points = qmc.scale(data_input[column].values, lower_boundaries, upper_boundaries, reverse = True)

            # if sample is None:
            #     sample = unscaled_points
            # else:
            #     sample = np.vstack((sample, unscaled_points))
            # starting_point = i + 1
        else:
            break

    """
    for i in range(starting_point, num_rounds + 1):
                
        if i == 1:

            sampler = qmc.LatinHypercube(d = len(lower_boundaries), optimization = 'random-cd')
            sample = sampler.random(n = n_samples)
            sliced_unscaled_points = sample

            print(f'Unprogressed round {i} discrepancy:', qmc.discrepancy(sample))

        else:

            eLHS = ExpandLHS(sample)

            sample = eLHS(n_samples, optimize = 'discrepancy')

            print(f'Progressed sample {i} discrepancy:', qmc.discrepancy(sample))

            sliced_unscaled_points = sample[-n_samples:]


        round_sample_scaled = qmc.scale(sliced_unscaled_points, lower_boundaries, upper_boundaries)

        df = pd.DataFrame(round_sample_scaled, columns = column)

        #             df['t_STAR'] = 0.5
        # df['L_X'] = 40.5
        # df['NU_X_THRESH'] = 500.0
        # df['X_RAY_SPEC_INDEX'] = 1.0

        df['Round'] = i
        df.to_hdf(f'GeneratedData/Input/{label}/{path}_r{i}.h5', mode = 'w', key = 'Data')

        round_points.append(df)
    """

    all_points = pd.concat(round_points, ignore_index = True)


    return all_points


def get_output(num_rounds, label):

    if label == 'TrainingData':
        path_out = 'training_data_output_2986'
        path_in = 'training_data_input_2986'

    elif label == 'ValidationData':
        path_in = 'validation_data_input_746'
        path_out = 'validation_data_output_746'

    else:
        path_in = 'test_data_input_933'
        path_out = 'test_data_output_933'

    all_outputs = {}

    
    starting_round = 1

    for i in range(1, num_rounds + 1):
        if os.path.exists(f'GeneratedData/Output/{label}/{path_out}_r{i}.h5'):
            print(f'Loading emulated round {i}')
    
            with h5py.File(f'GeneratedData/Output/{label}/{path_out}_r{i}.h5', 'r') as hf:

                for attr_name in hf.keys():
                    if attr_name not in all_outputs:  # load in the previous output data to not have to redo emulating
                        all_outputs[attr_name] = [] # here we attribute first instance of keys
                    all_outputs[attr_name].append(hf[attr_name][:])  # if not first instance, we just append them to the old keys

            starting_round = i + 1

        else:
            break
    
    """ 
    collect_outputs = {}

    for i in range(starting_round, num_rounds + 1):

        if not os.path.exists(f'GeneratedData/Input/{label}/{path_in}_r{i}.h5'):  # check if we have a file
            break

        data_input = pd.read_hdf(f'GeneratedData/Input/{label}/{path_in}_r{i}.h5')  # load in the input data to be emulated
        dropped_round = data_input.drop(['Round'], axis = 1)
        input_data = dropped_round.to_dict('records')

        collect_outputs = {}
        start_idx = 0


        while len(input_data) - start_idx > 0:

            end_idx = min(len(input_data), start_idx + batch_size)
            inputs = input_data[start_idx:end_idx] 

            normed_input_params, output, output_errors = emu.predict(inputs, verbose = True)


            if not collect_outputs:  # here we get the different attribues of the output and add the relevant ones to a dictionary with empty list
                for attr_name in dir(output):
                    attr_value = getattr(output, attr_name)
                    if not attr_name.startswith('_') and isinstance(attr_value, np.ndarray):
                        collect_outputs[attr_name] = []

            for attr_name in collect_outputs.keys():  # here we take the values of the relevant attributes and put them as values to the correct keys in teh dictionary
                collect_outputs[attr_name].append(getattr(output, attr_name))


            start_idx += batch_size


        combine = {}
        for attr_name, array_list in collect_outputs.items():
            combine[attr_name] = np.concatenate(array_list, axis = 0)  # here we merge the values from the different output rounds


        with h5py.File(f'GeneratedData/Output/{label}/{path_out}_r{i}.h5', 'w') as hf:  # saving that rounds emulation data
            for attr_name, array_data in combine.items():
                hf.create_dataset(attr_name, data = array_data)
            print(f'Saved emulated round {i}')


        for attr_name, array_data in combine.items():  # add this rounds data to the master dictionary
            if attr_name not in all_outputs:
                all_outputs[attr_name] = []
            all_outputs[attr_name].append(array_data)
    """
    everything_merged = {}
    for attr_name, array_data in all_outputs.items():
        everything_merged[attr_name] = np.concatenate(array_data, axis = 0)

    output = SimpleNamespace(**everything_merged)  # here we make it so that we can use out.PS


    return output

def get_21cmoutput(label):

    emu_redshifts = [ 5.90059,   6.038602,  6.179374  ,6.322961  ,6.46942   ,6.618808 , 6.771184,
  6.926608  ,7.08514   ,7.246843,  7.411779  ,7.580015  ,7.751615  ,7.926647,
  8.10518   ,8.287283  ,8.473028,  8.662489  ,8.855739  ,9.052854  ,9.25391,
  9.458988  ,9.668168  ,9.881531, 10.09916  ,10.32114  ,10.54757  ,10.77852,
 11.01409  ,11.25437  ,11.49946, 11.74944  ,12.00443  ,12.26452  ,12.52981,
 12.80041  ,13.07641  ,13.35794,  13.6451   ,13.938    ,14.23676  ,14.5415,
 14.85233  ,15.16937  ,15.49276 , 15.82262  ,16.15907  ,16.50225  ,16.85229,
 17.20934  ,17.57353  ,17.945   , 18.3239   ,18.71038  ,19.10458  ,19.50668,
 19.91681  ,20.33514  ,20.76185  ,21.19708]
    
    real_k_values = np.array([0.03720577, 0.03920577, 0.07529437, 0.08026625000000001, 0.12300975, 0.14641164999999998, 0.21662555000000003, 0.27229464999999997, 
0.38765215, 0.50454325, 0.6998539500000001, 0.92651025, 1.26932375, 1.54078924, 2.1156804])
    
    props = emulator_properties()
    props.user_params['HII_DIM'] = 256
    props.user_params['DIM'] = 768
    props.user_params['N_THREADS'] = 60

    if label == 'TrainingData':
        path_out = 'training_data_output_2986'
        path_in = 'training_data_input_2986_FAST'

    elif label == 'ValidationData':
        path_in = 'validation_data_input_746'
        path_out = 'validation_data_output_746_FAST'

    else:
        path_in = 'test_data_input_933'
        path_out = 'test_data_output_933_FAST'

    out_file_path = f'GeneratedData/Output/{label}/{path_out}.h5'
    in_file_path = f'GeneratedData/Input/{label}/{path_in}.h5'


    
    if os.path.exists(out_file_path):
        print(f'Loading existing data for {label}')
        with h5py.File(out_file_path, 'r') as hf:
            ps_array = hf['PS'][:]
            k_array = hf['k'][:]
        return ps_array, k_array
    
    data_input = pd.read_hdf(in_file_path)
    
    # Drop 'Round' if it exists in the dataframe, ignore errors if it doesn't
    dropped_round = data_input.drop(['Round'], axis=1, errors='ignore')
    input_data = dropped_round.to_dict('records')
    

    all_PS = []
    k_values = None

    for idx, input_dict in enumerate(input_data):
        print(f"Processing sample {idx+1}/{len(input_data)}")
    
        ap = p21c.AstroParams(**input_dict)

        coeval = p21c.run_coeval(redshift = emu_redshifts,
            user_params = props.user_params,
            astro_params = ap,
            flag_options = props.flag_options,
            cosmo_params = props.cosmo_params,
            write = False,
            random_seed = 12345,
            cleanup = True,
            direc = '_cache'
            )
        
        single_PS = []

        for box in coeval:
            PS, k = tools21cm.power_spectrum.power_spectrum_1d(box.brightness_temp, box_dims= props.user_params['BOX_LEN'], kbins=real_k_values)
            single_PS.append(PS)
            k_values = k

        single_PS = np.array(single_PS)

        all_PS.append(single_PS)
    
    master_PS_array = np.array(all_PS) * (k_values)**3 / (2 * np.pi**2)

    with h5py.File(out_file_path, 'w') as hf:
        hf.create_dataset('PS', data = master_PS_array)
        hf.create_dataset('k', data = k_values)
        print(f'Saved generated data to {out_file_path}')



    return master_PS_array, k_values


def get_unique(data):

    data.k = np.unique(data.k)
    data.PS_redshifts = np.unique(data.PS_redshifts)
    data.redshifts = np.unique(data.redshifts)
    data.PS_ks = np.unique(data.PS_ks)
    data.Muv = np.unique(data.Muv)
    data.UVLF_redshifts = np.unique(data.UVLF_redshifts)

    return data


def low_PS(data, k_cut, method, eta):

    #Gaussian

    if method == 'Gaussian':

        return data.PS * np.exp( - np.power(data.k / k_cut, 2))


    #Sharp cut

    elif method == 'SharpCut':

        low_train_sharp = np.zeros_like(data.PS) + 1e-12
        for idx, k_value in enumerate(data.k):
            if k_value < k_cut:
                low_train_sharp[:, :, idx] = data.PS[:, :, idx]
            else:
                break
        
        return low_train_sharp


    #Soft transition (sigmoid)

    else: 

        return data.PS / (1 + np.power(data.k / k_cut, eta))


def plotting_Wk_vs_k(data, k_cut, eta):

    gauss = np.exp( - np.power(data.k / k_cut, 2))

    sharp = np.zeros_like(data.k)
    for idx, k_value in enumerate(data.k):
        if k_value < k_cut:
            sharp[idx] = 1
        else:
            break

    sigmoid = 1 / (1 + np.power(data.k / k_cut, eta))

    plt.plot(data.k, gauss, color = 'b', label = 'Gaussian')
    plt.scatter(data.k, gauss, color = 'b')
    plt.plot(data.k, sharp, color = 'r', label = 'Sharp Cut')
    plt.scatter(data.k, sharp, color = 'r')
    plt.plot(data.k, sigmoid, color = 'g', label = 'Sigmoid')
    plt.scatter(data.k, sigmoid, color = 'g')
    plt.ylabel('W(k)')
    plt.xlabel(r'k (Mpc$^{-1}$)')
    plt.xlim(data.k[0] - 1e-2, data.k[-1] + 1e-2)
    plt.legend()
    plt.title('W(k) vs k')
    plt.savefig('W(k) vs k', dpi = 300)
    plt.show()


def corner_plot(dataframe, title, filename, samples = None):
    
    colors = [
    'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
    'olive', 'cyan', 'magenta', 'yellow', 'teal', 'navy', 'maroon', 'lime',
    'indigo', 'turquoise', 'violet', 'gold', 'coral', 'salmon', 'khaki',
    'plum', 'orchid', 'crimson', 'tomato', 'sienna', 'chocolate', 'peru',
    'tan', 'darkgreen', 'darkblue'
]
    
    latex_mapping = {
        'ALPHA_STAR': r'$\alpha_{\star}$',
        'ALPHA_ESC': r'$\alpha_{\mathrm{esc}}$',
        'F_STAR10': r'$f_{\star,10}$',
        'F_ESC10': r'$f_{\mathrm{esc},10}$',
        'M_TURN': r'$M_{\mathrm{turn}}$',
        't_STAR': r'$t_{\star}$',
        'L_X' : r'$L_{X}$',
        "NU_X_THRESH" : r'$E_0$',
        "X_RAY_SPEC_INDEX" : r'$\alpha_{X}$'
        }

    dataframe = dataframe.rename(columns=latex_mapping)
    palette = []

    sns.set(style = "ticks", font_scale = 1.5)

    if samples is not None:
        dataframe = dataframe.sample(n = samples)

    unique_types = dataframe['Round'].nunique()

    for i in range(unique_types):
        palette.append(colors[i])

    g = sns.pairplot(dataframe, hue = 'Round', markers = 'o', palette = palette, plot_kws={"s": 40}, corner = True, diag_kind= None)
    g.fig.suptitle(title, y = 0.85)

    for i, ax_row in enumerate(g.axes):
        for j, ax in enumerate(ax_row):
            if ax:
                if i == j:
                    ax.set_visible(False)
                    continue
                ax.tick_params(direction = 'in', top = True, right = True, length = 4, width = 2, colors = 'black')
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
    plt.savefig(filename, dpi = 300, bbox_inches = "tight")
    plt.show()


def plotting_PS(true_data, emulated_data_sig, emulated_data_sharp, varying, size, k, z, filename):

    np.random.seed(123)

    rand = np.random.randint(0, len(true_data.PS), size = size)
    

    fs = 20

    cs = [
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'magenta', 'yellow', 'teal', 'navy', 'maroon', 'lime',
        'indigo', 'turquoise', 'violet', 'gold', 'coral', 'salmon', 'khaki',
        'plum', 'orchid', 'crimson', 'tomato', 'sienna', 'chocolate', 'peru',
        'tan', 'darkgreen', 'darkblue'
    ]

    lines = ['-', '--', '-.', ':', (0, (5, 10))]

    if varying == 'zs':
        

        for i, l in zip(rand, lines):

            plt.plot(true_data.PS_redshifts, true_data.PS[i, :, k], lw = 2, ls = l, color = 'black', alpha = 0.6)
            plt.plot(true_data.PS_redshifts, emulated_data_sig[i, :, k], lw = 2, ls = l, color = 'blue', alpha = 0.6)
            plt.plot(true_data.PS_redshifts, emulated_data_sharp[i, :, k], lw = 2, ls = l, color = 'red', alpha = 0.6)
        
        plt.ylabel(r'$\Delta_{21}^2$ [mk$^2$]', fontsize = fs)
        plt.xlabel(r'Redshift z', fontsize = fs)
        plt.xlim(true_data.PS_redshifts[0] - 0.1, true_data.PS_redshifts[-1] + 0.1)
        plt.yticks(fontsize = fs)
        plt.xticks(fontsize = fs)
        plt.yscale('log')
        plt.title('21cmEMU Power Spectrum', fontsize = fs)
        plt.tight_layout()
        plt.savefig(filename, dpi = 300)
        plt.show()
            
    else:

        for i, l in zip(rand, lines):

            plt.plot(true_data.k, true_data.PS[i, z, :], lw = 2, ls = l, color = 'black', alpha = 0.6)
            plt.plot(true_data.k, emulated_data_sig[i, z, :], lw = 2, ls = l, color = 'blue', alpha = 0.6)
            plt.plot(true_data.k, emulated_data_sharp[i, z, :], lw = 2, ls = l, color = 'red', alpha = 0.6)
    
        plt.ylabel(r'$\Delta_{21}^2$ [mk$^2$]', fontsize = fs)
        plt.xlabel(r'k (Mpc$^{-1}$)', fontsize = fs)
        plt.xlim(true_data.k[0] - 1e-2, true_data.k[-1] + 1e-2)
        plt.ylim(bottom = 2e-3)
        plt.yticks(fontsize = fs)
        plt.xticks(fontsize = fs)
        plt.yscale('log')
        plt.title('21cmEMU Power Spectrum', fontsize = fs)
        plt.tight_layout()
        plt.savefig(filename, dpi = 300)
        plt.show()


class EarlyStopping:
    def __init__(self, patience = 5, delta = 0, verbose = False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss * (1.0 - self.delta):
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else: 
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


class PSNN(nn.Module):
    def __init__(self, input_dim, layers):
        super().__init__()

        network = []
        current_dim = input_dim

        for hidden_dim in layers:
            network.append(nn.Linear(current_dim, hidden_dim))
            network.append(nn.LayerNorm(hidden_dim))
            network.append(nn.ReLU())
            network.append(nn.Dropout(0.2))
            current_dim = hidden_dim

        network.append(nn.Linear(current_dim, 720))
        self.net = nn.Sequential(*network)

    def forward(self, x): 

        output = self.net(x)

        PS_2D = output.view(-1, 60, 12)
        
        return PS_2D
    

def save_file(label, n_rounds, final_output):

    if label == 'TestData':
        path = 'test_data_output_933'

    elif label == 'ValidationData':
        path = 'validation_data_output_748'

    else:
        path = 'training_data_output_2986'

    filename = f'GeneratedData/Output/{label}/{path}_rounds_{n_rounds}.h5'

    with h5py.File(filename, 'w') as hf:

        for attr_name, array_data in vars(final_output).items():

            hf.create_dataset(attr_name, data = array_data)


def get_file(label, n_rounds):


    if label == 'TrainingData':
        path = 'training_data_output_2986'

    elif label == 'TestData':
        path = 'test_data_output_933'
    
    else:
        path = 'validation_data_output_748'

    output_dict = {}

    with h5py.File(f'GeneratedData/Output/{label}/{path}_rounds_{n_rounds}.h5', 'r') as hf:

        for key in hf.keys():
            output_dict[key] = hf[key][:]
        
    output = SimpleNamespace(**output_dict)

    return output
