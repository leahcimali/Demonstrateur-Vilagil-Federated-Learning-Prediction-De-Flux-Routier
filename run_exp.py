import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.utils_graph import create_graph, subgraph_dijkstra 
from src.utils_data import load_PeMS04_flow_data, TimeSeriesDataset, my_data_loader, createLoaders, preprocess_PeMS_data, plot_prediction
from src.models import LSTMModel, train_model, testmodel 
from src.fedutil import local_dataset, fed_training_plan
from src.metrics import calculate_metrics
import networkx as nx
import sys
import json

# Set the random seed
seed = 42
torch.manual_seed(seed)

# Get the path to the configuration file from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 ./run_exp.py CONFIG_FILE_PATH")
    sys.exit(1)
config_file_path = sys.argv[1]

# Load the configuration file using the provided path
with open(config_file_path) as f:
    config = json.load(f)

init_node = config['init_node']
n_neighbours = config['n_neighbours']
smooth = config['smooth']
center_and_reduce = config['center_and_reduce']
normalize = config['normalize']
sort_by_mean = config['sort_by_mean']
goodnodes = config['goodnodes']  #[118,168,261]
number_of_nodes =  config['number_of_nodes']  #[118,168,261]
window_size = config['window_size']
target_size = config['target_size']
stride = config['stride']
communication_rounds = config['communication_rounds']
local_epochs = config['local_epochs']
fed_epochs = config['fed_epochs']
learning_rate = config['learning_rate']

#Load traffic flow dataframe and graph dataframe from PEMS
PeMS, distance = load_PeMS04_flow_data()
PeMS, adjmat, meanstd_dict = preprocess_PeMS_data(PeMS, distance, init_node, n_neighbours, smooth, center_and_reduce, normalize, sort_by_mean)
datadict = local_dataset(PeMS[goodnodes],3,len(PeMS[goodnodes]),window_size=window_size,stride=stride, target_size=target_size)

if local_epochs:
    # Local Training 
    train_losses = {}
    val_losses = {}

    for j in range(3):
        local_model = LSTMModel(input_size=1, hidden_size=32, num_layers=6, output_size=1)
        data_dict = datadict[j]
        local_model, train_losses[j], val_losses[j] = train_model(local_model, data_dict['train'], data_dict['val'], model_path ='./local{}.pth'.format(j), num_epochs=local_epochs, remove = False, learning_rate=0.001)

# Federated Learning Experiment
if fed_epochs:
    main_model = LSTMModel(1,32,1)
    fed_training_plan(main_model, datadict, communication_rounds, fed_epochs)

