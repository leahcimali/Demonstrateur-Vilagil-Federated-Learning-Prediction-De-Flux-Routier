###############################################################################
# Libraries
###############################################################################
import glob

import json
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


@st.cache_data
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

#######################################################################
# Main
#######################################################################
st.header("General Statistics")

experiments = "experiments"
if path_files := glob.glob(f"./{experiments}/**/config.json", recursive=True):

    params_config_use_for_select = \
    [
        "time_serie_percentage_length",
        "number_of_nodes",
        "window_size",
        "prediction_horizon",
        "model"
    ]
    user_selection = map_path_experiments_to_params(path_files, params_config_use_for_select)

    path_experiment_selected = selection_of_experiment(user_selection)

    with open(f"{path_experiment_selected}/test.json") as f:
        results = json.load(f)
    with open(f"{path_experiment_selected}/config.json") as f:
        config = json.load(f)


    nodes = results.keys()
    mapping_captor_and_node = {}
    for node in results.keys():
        mapping_captor_and_node[config["nodes_to_filter"][int(node)]] = node

    df_federated_node = []
    for captor in mapping_captor_and_node.keys():
        if "Federated" in results[mapping_captor_and_node[captor]].keys():
            federated_node = results[mapping_captor_and_node[captor]]["Federated"]
            df_federated_node.append(federated_node)

    if df_federated_node != []:
        st.subheader("How much the Federated version is performant on average taking in account all the captors for the calculation of the statistics")
        df_federated_node = pd.DataFrame(df_federated_node)
        global_stats_fed_ver = df_federated_node.describe().T
        global_stats_fed_ver.rename(columns={'count': 'Nb captors'}, inplace=True)
        st.table(global_stats_fed_ver.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]))

        c1_boxplot_fed, c2_bobxplot_fed, c3_boxplot_fed = st.columns((1,2,1))
        fig, ax = plt.subplots()
        bar_plot_results = df_federated_node
        bar_plot_results.boxplot(column=["RMSE", "MAE"], ylabel="values", xlabel="Captor", ax=ax)
        plt.yticks(np.arange(0, max(bar_plot_results["RMSE"].max(), bar_plot_results["MAE"].max()), 10))
        with c2_bobxplot_fed:
            st.pyplot(fig, use_container_width=True)