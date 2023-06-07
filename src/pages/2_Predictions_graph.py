###############################################################################
# Libraries
###############################################################################
from os import path

import glob
import json
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px


from metrics import rmse
from config import Params


@st.cache_data
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
def load_numpy(path):
    return np.load(path)


def plot_slider(experiment_path):
    test_set = load_numpy(f"{params.save_model_path}/test_data_{mapping_captor_with_nodes[captor]}.npy")

    y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_captor_with_nodes[captor]}.npy")
    y_pred = load_numpy(f"{experiment_path}/y_pred_local_{mapping_captor_with_nodes[captor]}.npy")
    y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_captor_with_nodes[captor]}.npy")
    
    index = load_numpy(f"{params.save_model_path}/index_{mapping_captor_with_nodes[captor]}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')
    
    slider = st.slider('Select time?', 0, len(index)-params.prediction_horizon-params.window_size, params.prediction_horizon)

    def plot_prediction_graph(color, label, title, y_pred, i):
        df = pd.DataFrame({'Time': index[i:i + params.window_size + params.prediction_horizon], 'Traffic Flow': test_set[i:i + params.window_size + params.prediction_horizon].flatten()})
        df['Window'] = df['Traffic Flow'].where((df['Time'] >= index[i]) & (df['Time'] <= index[i + params.window_size - 1]))
        df['y_true'] = df['Traffic Flow'].where(df['Time'] >= index[i + params.window_size - 1])
        df[f'y_pred_{label}'] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred[i, :]])
        fig = px.line(df, x='Time', y=["Window"], color_discrete_sequence=['black'])
        fig.add_scatter(x=df['Time'], y=df['y_true'], mode='markers+lines', marker=dict(color='blue'), name='y_true')
        fig.add_scatter(x=df['Time'], y=df[f'y_pred_{label}'], mode='markers+lines', marker=dict(color=color), name=f'{label} RMSE : {rmse(y_true[i, :].flatten(), y_pred[i, :].flatten()):.2f}')
        fig.add_bar(x=df['Time'], y=(np.abs(df[f'y_pred_{label}'] - df['y_true'])), name='Absolute Error')
        fig.update_xaxes(tickformat='%H:%M', dtick=3600000)
        fig.update_layout(xaxis_title='Time', yaxis_title='Traffic Flow', title=f"{title} ({index[slider].strftime('%Y-%m-%d')})", title_font=dict(size=28), legend=dict(title='Legends', font=dict(size=16)))
        fig.add_vrect(x0=index[i], x1=index[i + params.window_size - 1], fillcolor='gray', opacity=0.2, line_width=0)
        return fig

    # FEDERATED
    fed_fig = plot_prediction_graph('green', 'Federated', "Federated Prediction", y_pred_fed, slider)
    
    # LOCAL
    local_fig = plot_prediction_graph('red', 'Local', "Local Prediction", y_pred, slider)
    
    with st.spinner('Plotting...'):
        st.plotly_chart(fed_fig, use_container_width=True)
        st.plotly_chart(local_fig, use_container_width=True)



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
st.header("Predictions Graph")

experiments = "experiments" # PATH where your experiments are saved
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


    mapping_captor_with_nodes = {}
    for node in results.keys():
        mapping_captor_with_nodes[config["nodes_to_filter"][int(node)]] = node


    if 'captor' not in st.session_state:
        st.session_state['captor'] = 0
    captor = st.selectbox('Choose the captor', mapping_captor_with_nodes.keys(), index=st.session_state['captor'])
    st.session_state['captor'] = int(mapping_captor_with_nodes[captor])

    metrics = list(results[mapping_captor_with_nodes[captor]]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    local_node = []
    if "local_only" in results[mapping_captor_with_nodes[captor]].keys():
        local_node = results[mapping_captor_with_nodes[captor]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=multiselect_metrics, index=["Captor alone"])

    federated_node = []
    if "Federated" in results[mapping_captor_with_nodes[captor]].keys():
        federated_node = results[mapping_captor_with_nodes[captor]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=multiselect_metrics, index=["Captor in Federation"])


    st.subheader("Captor in Federation vs Captor alone")
    fed_local_node = pd.concat((federated_node, local_node), axis=0)
    st.table(fed_local_node.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]).format("{:.2f}"))


    params = Params(f'{path_experiment_selected}/config.json')
    if (path.exists(f'{params.save_model_path}y_true_local_{mapping_captor_with_nodes[captor]}.npy') and
        path.exists(f"{path_experiment_selected}/y_pred_fed_{mapping_captor_with_nodes[captor]}.npy")):
        plot_slider(path_experiment_selected)
