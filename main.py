
import torch

from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, local_dataset, plot_prediction
from src.models import LSTMModel
from src.utils_training import train_model, testmodel
from src.utils_fed import fed_training_plan
from src.metrics import calculate_metrics
import src.config
import sys



seed = 42
torch.manual_seed(seed)


# Get the path to the configuration file from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 experiment.py CONFIG_FILE_PATH")
    sys.exit(1)
config_file_path = sys.argv[1]

params = src.config.Params(config_file_path)

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
if params.num_epochs_local_no_federation:
    # Local Training 
    train_losses = {}
    val_losses = {}

    for node in range(params.number_of_nodes):
        local_model = LSTMModel(input_size= LSTM_input_size,
                                hidden_size= LSTM_hidden_size,
                                num_layers= LSTM_num_layers,
                                output_size= LSTM_output_size)
    
        data_dict = datadict[node]
        local_model, train_losses[node], val_losses[node] = train_model(local_model, data_dict['train'], data_dict['val'], 
                                                                  model_path ='./local{}.pth'.format(node),
                                                                  num_epochs=params.num_epochs_local_no_federation, 
                                                                  remove = False, learning_rate=params.learning_rate)
        
        y_true, y_pred = testmodel(local_model,data_dict['test'], meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])
        print(calculate_metrics(y_true, y_pred))
        plot_prediction(y_true, y_pred, data_dict['test_data'],meanstd_dict[params.nodes_to_filter[node]], window_size =params.window_size , time_point_t=0, node=0)

# # Federated Learning Experiment
if params.num_epochs_local_federation:
    main_model = LSTMModel(input_size= LSTM_input_size,
                            hidden_size= LSTM_hidden_size,
                            num_layers= LSTM_num_layers,
                            output_size= LSTM_output_size)
    
    model = fed_training_plan(main_model, datadict, params.communication_rounds, params.num_epochs_local_federation)
for node in range(params.number_of_nodes):  
    y_true, y_pred = testmodel(local_model,data_dict['test'], meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])
    calculate_metrics(y_true, y_pred)
    plot_prediction(y_true, y_pred, data_dict['test_data'], meanstd_dict[params.nodes_to_filter[node]],  window_size =params.window_size , time_point_t=0, node=0)





