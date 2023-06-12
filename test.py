from os import makedirs
import torch
import importlib
import contextlib

from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, local_dataset, plot_prediction
from src.utils_training import testmodel
from src.metrics import calculate_metrics, metrics_table, Percentage_of_Superior_Predictions
import src.config
import sys
import numpy

import json

seed = 42
torch.manual_seed(seed)


# Get the path to the configuration file from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 experiment.py CONFIG_FILE_PATH")
    sys.exit(1)
config_file_path = sys.argv[1]

params = src.config.Params(config_file_path)

PATH_EXPERIMENTS = "experiments/" + params.save_model_path

makedirs(PATH_EXPERIMENTS, exist_ok=True)

with open(PATH_EXPERIMENTS +'test.txt', 'w') as f:
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
            datadict[node]['test_data'] = datadict[node]['test_data'] * meanstd_dict[params.nodes_to_filter[node]]['std'] + meanstd_dict[params.nodes_to_filter[node]]['mean']

            numpy.save(f"{PATH_EXPERIMENTS}test_data_{node}", datadict[node]['test_data'])
            numpy.save(f"{PATH_EXPERIMENTS}index_{node}", datadict[node]['test_data'].index)

            y_true, y_pred = testmodel(model(1,32,1), datadict[node]['test'], f'{PATH_EXPERIMENTS}local{node}.pth', meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])  
            local_metrics = calculate_metrics(y_true, y_pred)
            metrics_dict[node]['local_only'] = local_metrics
            numpy.save(f'{PATH_EXPERIMENTS}y_true_local_{node}', y_true)
            numpy.save(f'{PATH_EXPERIMENTS}y_pred_local_{node}', y_pred)

            y_true_fed, y_pred_fed = testmodel(model(1,32,1), datadict[node]['test'], f'{PATH_EXPERIMENTS}bestmodel_node{node}.pth', meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])
            fed_metrics = calculate_metrics(y_true_fed, y_pred_fed)
            metrics_dict[node][f'Federated'] = fed_metrics
            numpy.save(f'{PATH_EXPERIMENTS}y_true_fed_{node}', y_true_fed)
            numpy.save(f'{PATH_EXPERIMENTS}y_pred_fed_{node}', y_pred_fed)
            print(f'Federated vs local only for node {node} :')
            fed_metrics['Superior Pred %'], local_metrics['Superior Pred %'] = Percentage_of_Superior_Predictions(y_true, y_pred, y_true_fed, y_pred_fed)
            print(metrics_table({'Local' :local_metrics, f'Federated' : fed_metrics }))
            if params.plot :
                    plot_prediction(y_true, y_pred, datadict[node]['test_data'],meanstd_dict[params.nodes_to_filter[node]], window_size =params.window_size , time_point_t=params.time_point_to_plot, node=0, plot_fig_name = f'Localmodel_{params.num_epochs_local_no_federation}epochs_node_{node}' )
                    plot_prediction(y_true_fed, y_pred_fed, datadict[node]['test_data'],meanstd_dict[params.nodes_to_filter[node]], window_size =params.window_size , time_point_t=params.time_point_to_plot, node=0, plot_fig_name = f'Federatedmodel_{params.num_epochs_local_federation}epochs_node_{node}_round{params.communication_rounds}' )

with open(PATH_EXPERIMENTS + "test.json", "w") as outfile:
    json.dump(metrics_dict, outfile)

