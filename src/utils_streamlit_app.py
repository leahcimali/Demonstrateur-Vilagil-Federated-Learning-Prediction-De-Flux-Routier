import streamlit as st
import json
import numpy as np
import folium
import glob
from pathlib import PurePath


#  A dictionary to map the options to their aliases
#  Add here the parameters that you want the user use to filter among all the experiments
OPTION_ALIASES = {
    "time_serie_percentage_length": "%time series used",
    "number_of_nodes": "Choose the number of sensors",
    "window_size": "Choose the windows size",
    "prediction_horizon": "Choose how far you want to see in the future",
    "model": "Choose the model"
}


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
    | the length of the time series used {params.time_serie_percentage_length * 100}%"


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
    # folium.Circle(location=[lat + 0.0020, long + 0.0018], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    folium.Circle(location=[lat + 0.0020, long + 0.0018], color="black", radius=50, fill=True, opacity=1, fill_opacity=1, fill_color=color).add_to(map_folium)
    folium.map.Marker([lat + 0.0022, long + 0.0014], icon=folium.features.DivIcon(html=f"<div style='font-weight:bold; font-size: 15pt; color: black'>{int(value_percent * 100)}%</div>")).add_to(map_folium)
    # folium.Circle(location=[lat,long], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    # folium.Circle(location=[lat,long], color=color, radius=100*value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)


def get_color_fed_vs_local(fed_value, local_value, superior=True):
    red = "#fe7597"
    green = "#75ff5b"
    if (superior):
        return (green, red) if ((fed_value) >= (local_value)) else (red, green)
    return (green, red) if ((fed_value) < (local_value)) else (red, green)


def style_dataframe(df):
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


def switch_page(page_name: str):
    """@CREDIT: Zachary Blackwood

    Args:
        page_name (str):

    Raises:
        RerunException:
        ValueError:

    Returns:
        _type_:
    """
    from streamlit.runtime.scriptrunner import RerunData, RerunException
    from streamlit.source_util import get_pages

    def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")

    page_name = standardize_name(page_name)

    pages = get_pages("streamlit_app.py")  # OR whatever your main page is called

    for page_hash, config in pages.items():
        if standardize_name(config["page_name"]) == page_name:
            raise RerunException(
                RerunData(
                    page_script_hash=page_hash,
                    page_name=page_name,
                )
            )

    page_names = [standardize_name(config["page_name"]) for config in pages.values()]

    raise ValueError(f"Could not find page {page_name}. Must be one of {page_names}")