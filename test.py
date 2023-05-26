
import torch

from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, local_dataset, plot_prediction
from src.models import LSTMModel
from src.utils_training import train_model, testmodel
from src.utils_fed import fed_training_plan
from src.metrics import calculate_metrics, metrics_table
import src.config
import sys

import contextlib
import json 

seed = 42
torch.manual_seed(seed)


# Get the path to the configuration file from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 experiment.py CONFIG_FILE_PATH")
    sys.exit(1)
config_file_path = sys.argv[1]

params = src.config.Params(config_file_path)

with open(params.model_path +'test.txt', 'w') as f:
    with contextlib.redirect_stdout(f):

        LSTM_input_size = 1
        LSTM_hidden_size = 32
        LSTM_num_layers = 6
        LSTM_output_size = 1


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
            y_true, y_pred = testmodel(LSTMModel(1,32,1), datadict[node]['val'], f'{params.model_path}local{node}.pth', meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])  
            local_metrics = calculate_metrics(y_true, y_pred)
            metrics_dict[node]['local_only'] = local_metrics
            for round in range(1, params.communication_rounds+1):
                y_true_fed, y_pred_fed = testmodel(LSTMModel(1,32,1), datadict[node]['val'], f'{params.model_path}model_round_{round}.pth', meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])
                fed_metric = calculate_metrics(y_true_fed, y_pred_fed)
                metrics_dict[node][f'fed_round_{round}'] = fed_metric
                print(f'Federated vs local only for node {node} :')
                print(metrics_table({'Local' :local_metrics, f'Federated Round {round}' : fed_metric }))

with open(params.model_path +"test.json", "w") as outfile:
    json.dump(metrics_dict, outfile)























