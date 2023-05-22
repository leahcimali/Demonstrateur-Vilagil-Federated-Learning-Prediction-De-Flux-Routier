
from src.utils_data import my_data_loader, createLoaders


import pandas as pd
import torch

from src.utils_training import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import copy

def setup_models(n_nodes, main_model):
    """
    Initialize dictionary with n_nodes models

    n_nodes : int
        Number of nodes in the federated learning framework

    main_model: torch.nn.Module
        Torch neural network model

    Returns
    -------
    Dictionary with "n_nodes" models intialized

    """
    return {k: copy.deepcopy(main_model) for k in range(n_nodes)}
    
def send_model(main_model, model_dict, number_of_nodes):

    """
    Send federated model to nodes

    Parameters
    ----------
    main_model : torch.nn.Module

    model_dict : Dict
        Dictionary with the models of the different nodes

    number_of_nodes : int
        number of nodes in federation
    """

    with torch.no_grad():
        for i in range(number_of_nodes):
            state_dict = copy.deepcopy(main_model.state_dict())
            model_dict[i].load_state_dict(state_dict)
    return model_dict

def fedavg(main_model, model_dict, number_of_nodes):

    """
    Implementation of the FedAvg Algorithm as described in:
    McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data. 
    Artificial intelligence and statistics. PMLR, 2017
    """

    state_dict = model_dict[0].state_dict()
    for name, param in model_dict[0].named_parameters():
        for i in range(1, number_of_nodes):
            state_dict[name]=  state_dict[name] + model_dict[i].state_dict()[name]
        state_dict[name] = state_dict[name]/number_of_nodes
    new_model = copy.deepcopy(main_model)
    new_model.load_state_dict(state_dict)
    return new_model




def fed_training_plan(main_model, data_dict, rounds=3, epoch=200, model_path= './'):
    
    """
    Controler function to launch federated learning
    
    Parameters
    ----------
    main_model: 
        Define the central node model :

    data_dict : Dictionary
      Contains training and validation data for the different FL nodes

    rounds : int
        Number of federated learning rounds

     
    epoch : int
        Number of training epochs in each round

    model_path : str
        Define the path where to save the models 
    
    """
    
    nodes = len(data_dict)
    
    model_dict = setup_models(nodes,main_model)
    
    for round in range(1,rounds+1):
    
        print('Init round {} :'.format(round))
    
        model_dict = send_model(main_model, model_dict, nodes)
    
        for i in range(nodes):
            print('Training node {} for round {}'.format(i, round))
            model_dict[i], _ , _ = train_model(model_dict[i], data_dict[i]['train'], data_dict[i]['val'], f'{model_path}local{i}_round{round}.pth', epoch, remove = True)
    
        print('FedAVG for round {}:'.format(round))
    
        main_model = fedavg(main_model, model_dict, nodes)
    
        print('Done')
        torch.save(main_model.state_dict(), f'{model_path}model_round_{round}.pth')
    
    print("FedAvg All Rounds Complete !")
    return main_model



