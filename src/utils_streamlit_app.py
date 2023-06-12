import streamlit as st
import json
import numpy as np
import folium


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


def format_windows_prediction_size(value):
    return f"{int((value * 5 ) / 60)}h (t+{(value)})"


def selection_of_experiment(possible_choice):
    """
    Create the visual for choosing an experiment

    Parameters:
    -----------


    Returns:
        return the path to the experiment that the user choose
    """
    st.subheader("Selection of experiment")
    st.markdown("""
                *The selection is sequential*
                """)

    time_serie_percentage_length = st.selectbox('Choose the time series length used to train the model', possible_choice["time_serie_percentage_length"].keys())

    nb_captor_filtered = filtering_path_file(possible_choice["number_of_nodes"], possible_choice["time_serie_percentage_length"][time_serie_percentage_length])
    nb_captor = st.selectbox('Choose the number of sensors', nb_captor_filtered.keys())

    windows_size_filtered = filtering_path_file(possible_choice["window_size"], possible_choice["number_of_nodes"][nb_captor])
    window_size = st.selectbox('Choose the windows size', windows_size_filtered.keys(), format_func=format_windows_prediction_size)

    horizon_filtered = filtering_path_file(possible_choice["prediction_horizon"], windows_size_filtered[window_size])
    horizon_size = st.selectbox('Choose how far you want to see in the future', horizon_filtered.keys(), format_func=format_windows_prediction_size)

    models_filtered = filtering_path_file(possible_choice["model"], horizon_filtered[horizon_size])
    model = st.selectbox('Choose the model', models_filtered.keys())

    if len(models_filtered[model]) <= 1:
        return ("\\".join(models_filtered[model][0].split("\\")[:-1]))

    st.write("TODO : WARNING ! More than one results correspond to your research pick only one (see below)")
    select_exp = st.selectbox("Choose", models_filtered[model])
    select_exp = models_filtered[model].index(select_exp)
    return ("\\".join(models_filtered[model][select_exp].split("\\")[:-1]))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


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
    folium.Circle(location=[lat + 0.0020, long + 0.0018], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    folium.Circle(location=[lat + 0.0020, long + 0.0018], color=color, radius=100 * value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)
    folium.map.Marker([lat + 0.0022, long + 0.0014], icon=folium.features.DivIcon(html=f"<div style='font-weight:bold; font-size: 15pt; color: black'>{int(value_percent * 100)}%</div>")).add_to(map_folium)
    # folium.Circle(location=[lat,long], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    # folium.Circle(location=[lat,long], color=color, radius=100*value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)


def get_color_fed_vs_local(fed_value, local_value):
    red = "#fe7597"
    green = "#7cff2d"
    return (green, red) if ((fed_value) >= (local_value)) else (red, green)
