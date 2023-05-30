from os import makedirs
import torch
import importlib
import contextlib

from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, local_dataset, plot_prediction
from src.utils_training import train_model, testmodel
from src.utils_fed import fed_training_plan
from src.metrics import calculate_metrics, metrics_table
import src.config
import sys

import json 

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

with open(params.save_model_path +'test.txt', 'w') as f:
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
        metrics_dict ={}
        for node in range(len(params.nodes_to_filter)):
            metrics_dict[node]={}
            y_true, y_pred = testmodel(model(1,32,1), datadict[node]['test'], f'{params.save_model_path}local{node}.pth', meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])  
            local_metrics = calculate_metrics(y_true, y_pred)
            metrics_dict[node]['local_only'] = local_metrics

            y_true_fed, y_pred_fed = testmodel(model(1,32,1), datadict[node]['test'], f'{params.save_model_path}bestmodel_node{node}.pth', meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])
            fed_metric = calculate_metrics(y_true_fed, y_pred_fed)
            metrics_dict[node][f'fed_round_{round}'] = fed_metric
            print(f'Federated vs local only for node {node} :')
            print(metrics_table({'Local' :local_metrics, f'Federated' : fed_metric }))

with open(params.save_model_path + "test.json", "w") as outfile:
    json.dump(metrics_dict, outfile)

for node in range(params.number_of_nodes):  
                if params.plot : 
                    plot_prediction(y_true, y_pred, datadict['test_data'],meanstd_dict[params.nodes_to_filter[node]], window_size =params.window_size , time_point_t=params.time_point_to_plot, node=0, plot_fig_name = f'Federated_{params.num_epochs_local_federation}epochs_node_{node}_round_{params.communication_rounds}' )





















