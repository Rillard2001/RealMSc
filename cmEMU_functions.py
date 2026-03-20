import numpy as np
from scipy.stats import qmc
from expandLHS import ExpandLHS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace
import h5py


def lhs_sampler(num_rounds, label):

    #test_param = [-1.3, 0.5, -1.0, -0.5, 8.7, 0.5, 40.5, 500.0, 1.0]
    column = ['F_STAR10', 'ALPHA_STAR', 'F_ESC10', 'ALPHA_ESC', 'M_TURN']
        

    # lower_boundaries = [-3.0, -0.5, -3.0, -1.0, 8.0, 0.1, 38.0, 100.0, -1.0]
    # upper_boundaries = [-0.05, 1.0, -0.05, 0.5, 10.0, 1.0, 42.0, 1500.0, 3.0]

    lower_boundaries = [-3.0, -0.5, -3.0, -1.0, 8.0]
    upper_boundaries = [-0.05, 1.0, -0.05, 0.5, 10.0]



    if label == 'TrainingData':
        path = 'training_data_input_2986'
        n_samples = 2986

    elif label == 'ValidationData':
        path = 'validation_data_input_746'
        n_samples = 746

    else:
        path = 'test_data_input_933'
        n_samples = 933

    round_points = []
    starting_point = 1
    sample = None

    for i in range(1, num_rounds + 1):
        if os.path.exists(f'GeneratedData/Input/{label}/{path}_r{i}_4fixed.h5'):
            print('Loading')
            data_input = pd.read_hdf(f'GeneratedData/Input/{label}/{path}_r{i}_4fixed.h5')
            round_points.append(data_input)
            unscaled_points = qmc.scale(data_input[column].values, lower_boundaries, upper_boundaries, reverse = True)

            if sample is None:
                sample = unscaled_points
            else:
                sample = np.vstack((sample, unscaled_points))
            starting_point = i + 1
        else:
            break

    
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
        df['t_STAR'] = 0.5
        df['L_X'] = 40.5
        df['NU_X_THRESH'] = 500.0
        df['X_RAY_SPEC_INDEX'] = 1.0
        df['Round'] = i
        df.to_hdf(f'GeneratedData/Input/{label}/{path}_r{i}_4fixed.h5', mode = 'w', key = 'Data')

        round_points.append(df)


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
        if os.path.exists(f'GeneratedData/Output/{label}/{path_out}_r{i}_4fixed.h5'):
            print(f'Loading emulated round {i}')
    
            with h5py.File(f'GeneratedData/Output/{label}/{path_out}_r{i}_4fixed.h5', 'r') as hf:

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

        if not os.path.exists(f'GeneratedData/Input/{label}/{path_in}_r{i}_4fixed.h5'):  # check if we have a file
            break

        data_input = pd.read_hdf(f'GeneratedData/Input/{label}/{path_in}_r{i}_4fixed.h5')  # load in the input data to be emulated
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


        with h5py.File(f'GeneratedData/Output/{label}/{path_out}_r{i}_4fixed.h5', 'w') as hf:  # saving that rounds emulation data
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
    palette = []

    sns.set(style = "ticks")

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
    plt.savefig(filename, dpi = 300)
    plt.show()


def plotting_PS(true_data, low_true_data, emulated_data, varying, size, k, z, filename):

    np.random.seed(42)

    rand = np.random.randint(0, len(true_data.PS), size = size)
    

    fs = 20

    cs = [
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'magenta', 'yellow', 'teal', 'navy', 'maroon', 'lime',
        'indigo', 'turquoise', 'violet', 'gold', 'coral', 'salmon', 'khaki',
        'plum', 'orchid', 'crimson', 'tomato', 'sienna', 'chocolate', 'peru',
        'tan', 'darkgreen', 'darkblue'
    ]

    if varying == 'zs':

        for i, c in zip(rand, cs):

            plt.plot(true_data.PS_redshifts, true_data.PS[i, :, k], lw = 2, ls = '-', color = c)
            plt.plot(true_data.PS_redshifts, emulated_data[i, :, k], lw = 2, ls = '-.', color = c)
        
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

        for i, c in zip(rand, cs):

            plt.plot(true_data.k, true_data.PS[i, z, :], lw = 2, ls = '-', color = c)
            plt.plot(true_data.k, emulated_data[i, z, :], lw = 2, ls = '-.', color = c)
    
        plt.ylabel(r'$\Delta_{21}^2$ [mk$^2$]', fontsize = fs)
        plt.xlabel(r'k (Mpc$^{-1}$)', fontsize = fs)
        plt.xlim(true_data.k[0] - 1e-2, true_data.k[-1] + 1e-2)
        plt.ylim(2e-3, 6e3)
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