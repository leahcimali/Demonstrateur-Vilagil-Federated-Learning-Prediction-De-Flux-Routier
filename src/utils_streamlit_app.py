import streamlit as st
import json
import numpy as np
import pandas as pd
import folium
import glob
from pathlib import PurePath

from utils_data import load_PeMS04_flow_data
from utils_graph import create_graph


#  A dictionary to map the options to their aliases
#  Add here the parameters that you want the user use to filter among all the experiments
OPTION_ALIASES = {
    "time_serie_percentage_length": "%time series used",
    "number_of_nodes": "Choose the number of sensors",
    "window_size": "Choose the windows size",
    "prediction_horizon": "Choose how far you want to see in the future",
    "model": "Choose the model"
}

METRICS = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]


@st.cache_resource
def filtering_path_file(file_dict, filter_list):
    """
    Returns a new dictionary that contains only the files that are in the filter list.

    Parameters
    ----------
    file_dict : dict
        The dictionary that map each options with a list of files.
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


@st.cache_data
def load_experiment_results(experiment_path):
    with open(f"{experiment_path}/test.json") as f:
        results = json.load(f)
    return results


@st.cache_data
def load_experiment_config(experiment_path):
    with open(f"{experiment_path}/config.json") as f:
        config = json.load(f)
    return config


@st.cache_resource
def map_path_experiments_to_params(path_files, params_config_use_for_select):
    """
    Map all path experiments with the parameters of their config.json

    Parameters:
    -----------
        path_files :
            The path to the config.json of all experiments
        params_config_use_for_select :
            The parameters use for the selection

    Returns:
        The mapping between path to the experiment and the parameters of the experiement
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
            with open(file) as f:  # Load the config of an experiment
                config = json.load(f)
            if config[param] in mapping_path_with_param[param].keys():
                mapping_path_with_param[param][config[param]].append(file)  # Map the path to the experiment to a parameter and the value of the parameter (see exemple)
            else:
                mapping_path_with_param[param][config[param]] = [file]  # Map the path to the experiment to a parameter and the value of the parameter (see exemple)
    return mapping_path_with_param


def format_windows_prediction_size(value):
    return f"{int((value * 5 ) / 60)}h (t+{(value)})"


#  A function to convert the option to its alias
def format_option(option):
    return OPTION_ALIASES.get(option, option)


def format_radio(path_file_experiment):
    from config import Params
    params = Params(f'{path_file_experiment}')
    return f"{params.model} | Sensor(s): ({params.number_of_nodes}) {params.nodes_to_filter} \
    | prediction {format_windows_prediction_size(params.prediction_horizon)} \
    | the length of the time series used {params.time_serie_percentage_length * 100}% \
    | batch size use {params.batch_size}"


def selection_of_experiment():
    """
    Create the visual to choose an experiment

    Parameters:
    -----------

    Returns:
        return the path to the experiment that the user choose
    """

    experiments = "./experiments/"  # PATH where all the experiments are saved
    if path_files := glob.glob(f"./{experiments}**/config.json", recursive=True):

        options = list(OPTION_ALIASES.keys())
        map_path_experiments_params = map_path_experiments_to_params(path_files, options)

        selected_options = st.multiselect(
            "Choose the options you want to use to filter the experiments",
            options,
            format_func=format_option,
            default=["number_of_nodes", "prediction_horizon", "model"]
        )

        if (len(selected_options) > 0):
            selectbox_options = {}
            previous_option = None
            previous_path_file = None
            for option in selected_options:
                if previous_option is not None:
                    option_filtered = filtering_path_file(map_path_experiments_params[option], previous_path_file)
                else:
                    option_filtered = map_path_experiments_params[option]
                selectbox_options[option] = {
                    "select": st.selectbox(
                        format_option(option),
                        option_filtered.keys(),
                        format_func=format_windows_prediction_size
                        if option
                        in ["window_size", "prediction_horizon"]
                        else str,
                    )
                }
                previous_path_file = option_filtered[selectbox_options[option]["select"]]
                previous_option = option

            select_exp = st.radio("Choose", list(previous_path_file), format_func=format_radio)

            return PurePath(select_exp).parent
        return None


def selection_of_experiment_cluster():
    """
    Create the visual to choose a cluster experiment

    Parameters:
    -----------

    Returns:
        return the path to the experiment that the user choose
    """

    experiments_folder = "community_experiments"
    if experiments_path := glob.glob(f"./{experiments_folder}/*", recursive=True):
        select_exp = st.selectbox("Choose the experiment cluster", list(experiments_path))
        return PurePath(select_exp)
    else:
        return None


def create_selectbox_metrics():
    return st.selectbox("Choose the metric", METRICS)


def create_circle_precision_predict(marker_location, value_percent, map_folium, color):
    """
    Draw a circle at the position of the marker.

    Parameters:
    ----------
        marker_location (Marker Folium)

        value_percent (float)

        map_folium (Map Folium)

        color :
            Hex code HTML
    """
    lat, long = marker_location
    # folium.Circle(location=[lat + 0.0020, long + 0.0018], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    folium.Circle(location=[lat + 0.0020, long + 0.0018], color="black", radius=50, fill=True, opacity=1, fill_opacity=1, fill_color=color).add_to(map_folium)
    folium.map.Marker([lat + 0.0022, long + 0.0014], icon=folium.features.DivIcon(html=f"<div style='font-weight:bold; font-size: 15pt; color: black'>{int(value_percent * 100)}%</div>")).add_to(map_folium)
    # folium.Circle(location=[lat,long], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    # folium.Circle(location=[lat,long], color=color, radius=100*value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)


def get_color_fed_vs_local(fed_value, local_value, superior=True):
    red = "#fe4269"
    green = "#00dd00"
    fed_value = float(fed_value)
    local_value = float(local_value)
    if (superior):
        return (green, red) if ((fed_value) >= (local_value)) else (red, green)
    return (green, red) if ((fed_value) < (local_value)) else (red, green)


def style_dataframe(df, colors=None, column_index=None):
    styles = []
    for i in range(len(df)):
        if i % 2 == 0:
            styles.append({
                'selector': f'tbody tr:nth-child({i+1})',
                'props': [('background-color', 'rgba(200, 200, 200, 0.8)', ), ('color', 'black')],
            })
        else:
            styles.append({
                'selector': f'tbody tr:nth-child({i+1})',
                'props': [('background-color', 'rgba(230, 230, 230, 0.8)'), ('color', 'black')],
            })
        if (colors is not None and column_index is not None):
            styles.append({
                'selector': f'tbody tr:nth-child({i+1}) > :nth-child({column_index})',
                'props': [
                    ('font-weight', 'bold'),
                    ('color', f'{colors[i]}'),
                ],
            })
    styles.extend(
        (
            {
                'selector': 'th',
                'props': [('font-weight', 'bold'), ('color', 'black')],
            },
            {
                'selector': 'tbody tr > :nth-child(1)',
                'props': [
                    ('background-color', 'rgba(160, 100, 170, 0.3)'),
                    ('color', 'black'),
                ],
            },
        )
    )
    return styles


def results_to_stats_dataframe(results):
    df = pd.DataFrame(results, columns=METRICS)
    df = df.describe().T
    df.drop(columns={'count'}, inplace=True)
    df = df.applymap(lambda x: '{:.2f}'.format(x))
    return df


def get_results_for_key(results, sensors, key):
    return [
        results[sensor][key]
        for sensor in sensors
        if key in results[sensor].keys()
    ]


def get_colors_for_results(df_fed, df_local, columns):
    color_fed = []
    color_local = []
    for i in range(len(METRICS)):
        if (i < len(METRICS) - 1):  # because "Superior Pred %" metric needs to be superior=True
            col_fed, col_local = get_color_fed_vs_local(df_fed.iloc[i][columns], df_local.iloc[i][columns], superior=False)
        else:
            col_fed, col_local = get_color_fed_vs_local(df_fed.iloc[i][columns], df_local.iloc[i][columns], superior=True)
        color_fed.append(col_fed)
        color_local.append(col_local)
    return color_fed, color_local


def results_to_dataframe(results, sensor_selected, version):
    return pd.DataFrame(results[sensor_selected][version], columns=METRICS, index=["Value"]).T.applymap(lambda x: '{:.2f}'.format(x))


def get_name_version_normalized(normalized=True):
    if normalized:
        federated_ver = "Federated_normalized"
        local_only_ver = "local_only_normalized"
    else:
        federated_ver = "Federated_unormalized"
        local_only_ver = "local_only_unormalized"
    return federated_ver, local_only_ver


@st.cache_resource
def load_graph():
    _, distance = load_PeMS04_flow_data()
    return create_graph(distance)