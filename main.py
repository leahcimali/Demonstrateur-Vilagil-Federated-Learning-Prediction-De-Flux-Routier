
from os import makedirs
import torch
import importlib
import contextlib

from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, local_dataset, plot_prediction
from src.utils_training import train_model, testmodel
from src.utils_fed import fed_training_plan
from src.metrics import calculate_metrics
import src.config
import sys

import contextlib


seed = 42
torch.manual_seed(seed)


# Get the path to the configuration file from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 experiment.py CONFIG_FILE_PATH")
    sys.exit(1)
config_file_path = sys.argv[1]

params = src.config.Params(config_file_path)

path_folder = params.save_model_path.split("/")[:-1]
new_path = '/'.join(path_folder)
makedirs(new_path, exist_ok=True)

with open(params.save_model_path + 'train.txt', 'w') as f:
    with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):

        module_name = 'src.models'
        class_name = params.model
        module = importlib.import_module(module_name)
        model = getattr(module, class_name)

        input_size = 1
        hidden_size = 32
        num_layers = 6
        output_size = 1


        #Load traffic flow dataframe and graph dataframe from PEMS
        df_PeMS, distance = load_PeMS04_flow_data()
        df_PeMS = df_PeMS[:int(len(df_PeMS)*params.time_serie_percentage_length)]
        print(len(df_PeMS))
        df_PeMS, adjmat, meanstd_dict = preprocess_PeMS_data(df_PeMS, distance, params.init_node, params.n_neighbours,
                                                            params.smooth, params.center_and_reduce,
                                                            params.normalize, params.sort_by_mean)
        print(params.nodes_to_filter)
        datadict = local_dataset(df = df_PeMS,
                                nodes = params.nodes_to_filter,
                                window_size=params.window_size,
                                stride=params.stride,
                                prediction_horizon=params.prediction_horizon)
        print(datadict.keys())
        if params.num_epochs_local_no_federation:
            # Local Training 
            train_losses = {}
            val_losses = {}

            for node in range(params.number_of_nodes):
                local_model = model(input_size, hidden_size, output_size, num_layers)
            
                data_dict = datadict[node]
                local_model, train_losses[node], val_losses[node] = train_model(local_model, data_dict['train'], data_dict['val'], 
                                                                        model_path = f'{params.save_model_path}local{node}.pth',
                                                                        num_epochs=params.num_epochs_local_no_federation, 
                                                                        remove = False, learning_rate=params.learning_rate)
                
                y_true, y_pred = testmodel(local_model,data_dict['test'], meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])
                if params.plot : 
                    plot_prediction(y_true, y_pred, data_dict['test_data'],meanstd_dict[params.nodes_to_filter[node]], window_size =params.window_size , time_point_t=params.time_point_to_plot, node=0, plot_fig_name = f'Local_{params.num_epochs_local_no_federation}epochs_node_{node}' )

        # # Federated Learning Experiment
        if params.num_epochs_local_federation:
            main_model = model(input_size, hidden_size, output_size, num_layers)
            
            model = fed_training_plan(main_model, datadict, params.communication_rounds, params.num_epochs_local_federation, model_path = f'{params.save_model_path}',)
            






