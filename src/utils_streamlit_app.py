import streamlit as st
import json
import numpy as np
import pandas as pd

@st.cache_resource
def filtering_path_file(file_dict, filter_list):
    """
    Returns a new dictionary that contains only the files that are in the filter list.

    Parameters
    ----------
    file_dict : dict
        The dictionary that maps some keys to lists of files.
    filter_list : list
        The list of files that are used as a filter.

    Return
    ------
    filtered_file_dict : dict
        The new dictionary that contains only the keys and files from file_dict that are also in filter_list.
    """
    filtered_file_dict = {}
    for key in file_dict.keys():
        for file in file_dict[key]:
            if file in filter_list:
                if key in filtered_file_dict:
                    filtered_file_dict[key].append(file)
                else:
                    filtered_file_dict[key] = [file]
    return filtered_file_dict

@st.cache_resource
def load_numpy(path):
    return np.load(path)

@st.cache_resource
def map_path_experiments_to_params(path_files, params_config_use_for_select):
    """
    Map all path experiments with parameters of their config.json

    Parameters:
    -----------
        path_files :
            The path to the config.json of all experiments
        params_config_use_for_select :
            The parameters use for the selection

    Returns:
        The mapping between path to the experiments and parameters of the experiements
        exemple :
        mapping = {
            "nb_node" : {
                3: [path_1, path_3],
                4: [path_4]
            },
            windows_size: {
                1: [path_1],
                5: [path_3],
                8: [path_4]
            }
        }
    """
    mapping_path_with_param = {}
    for param in params_config_use_for_select:
        mapping_path_with_param[param] = {}
        for file in path_files:
            with open(file) as f:
                config = json.load(f)
            if config[param] in mapping_path_with_param[param].keys():
                mapping_path_with_param[param][config[param]].append(file)
            else:
                mapping_path_with_param[param][config[param]] = [file]
    return mapping_path_with_param

def selection_of_experiment(possible_choice):
    """
    Create the visual for choosing an experiment

    Parameters:
    -----------


    Returns:
        return the path to the experiment that the user choose
    """

    time_serie_percentage_length = st.selectbox('Choose the time series length', possible_choice["time_serie_percentage_length"].keys())

    nb_captor_filtered = filtering_path_file(possible_choice["number_of_nodes"], possible_choice["time_serie_percentage_length"][time_serie_percentage_length])
    nb_captor = st.selectbox('Choose the number of captor', nb_captor_filtered.keys())

    windows_size_filtered = filtering_path_file(possible_choice["window_size"], possible_choice["number_of_nodes"][nb_captor])
    window_size = st.selectbox('Choose the windows size', windows_size_filtered.keys())

    horizon_filtered = filtering_path_file(possible_choice["prediction_horizon"], windows_size_filtered[window_size])
    horizon_size = st.selectbox('Choose the prediction horizon', horizon_filtered.keys())

    models_filtered = filtering_path_file(possible_choice["model"], horizon_filtered[horizon_size])
    model = st.selectbox('Choose the model', models_filtered.keys())

    if(len(models_filtered[model]) > 1):
        st.write("TODO : WARNING ! More than one results correspond to your research pick only one (see below)")
        select_exp = st.selectbox("Choose", models_filtered[model])
        select_exp = models_filtered[model].index(select_exp)
        experiment_path = ("\\".join(models_filtered[model][select_exp].split("\\")[:-1]))

    else:
        experiment_path = ("\\".join(models_filtered[model][0].split("\\")[:-1]))

    return experiment_path