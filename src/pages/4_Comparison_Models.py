###############################################################################
# Libraries
###############################################################################
from os import path


import glob
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


from utils_streamlit_app import load_numpy, map_path_experiments_to_params, filtering_path_file
from config import Params

st.set_page_config(layout="wide")


def compute_absolute_error(path):
    y_true = load_numpy(f"{path}/y_true_local_{mapping_sensor_with_nodes_model_1[sensor_select]}.npy")
    y_pred_fed = load_numpy(f"{path}/y_pred_fed_{mapping_sensor_with_nodes_model_2[sensor_select]}.npy")
    return (np.abs(y_pred_fed.flatten() - y_true.flatten()))


def plot_slider(experiment_path):
    ae_model_1 = compute_absolute_error(experiment_path[0])
    ae_model_2 = compute_absolute_error(experiment_path[1])

    def plot_box(title, ae, max_y_value, color):
        fig = px.box(y=ae, color_discrete_sequence=[color], title=title, points="suspectedoutliers")
        fig.update_layout(
            title={
                'text': f"{title}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            yaxis_title="Trafic flow (absolute error)",
            yaxis=dict(range=[0, max_y_value]),
            font=dict(
                size=28,
                color="#7f7f7f"
            ), height=900, width=250
        )
        return fig

    max_y_value = max(max(ae_model_1), max(ae_model_2))

    title_model_1 = paths_experiment_selected[0].split("\\")[1]
    fig_model_1 = plot_box(f"{title_model_1}", ae_model_1, max_y_value, "green")

    title_model_2 = paths_experiment_selected[1].split("\\")[1]
    fig_model_2 = plot_box(f"{title_model_2}", ae_model_2, max_y_value, "red")

    with st.spinner('Plotting...'):
        st.subheader(f"Comparison between two models on sensor {sensor_select} on the federated version (Aboslute Error)")
        _, c2_fed_fig, c3_local_fig, _ = st.columns((1, 1, 1, 1))
        with c2_fed_fig:
            st.plotly_chart(fig_model_1, use_container_width=False)
        with c3_local_fig:
            st.plotly_chart(fig_model_2, use_container_width=False)


def selection_of_experiment(possible_choice):
    time_serie_percentage_length = st.selectbox('Choose the time series length', possible_choice["time_serie_percentage_length"].keys())

    nb_sensor_filtered = filtering_path_file(possible_choice["number_of_nodes"], possible_choice["time_serie_percentage_length"][time_serie_percentage_length])
    nb_sensor = st.selectbox('Choose the number of sensor_select', nb_sensor_filtered.keys())

    windows_size_filtered = filtering_path_file(possible_choice["window_size"], possible_choice["number_of_nodes"][nb_sensor])
    window_size = st.selectbox('Choose the windows size', windows_size_filtered.keys())

    horizon_filtered = filtering_path_file(possible_choice["prediction_horizon"], windows_size_filtered[window_size])
    horizon_size = st.selectbox('Choose the prediction horizon', horizon_filtered.keys())

    models_filtered = filtering_path_file(possible_choice["model"], horizon_filtered[horizon_size])

    col1_model_1, col2_model_2 = st.columns(2)
    with col1_model_1:
        model_1 = st.radio(
            "Choose the first model",
            models_filtered.values(), key="model_1")
    with col2_model_2:
        model_2 = st.radio(
            "Choose the second model",
            models_filtered.values(), key="model_2")

    return [model_1[0], model_2[0]]


#######################################################################
# Main
#######################################################################


st.header("Comparison Models")

experiments = "experiments"  # PATH where your experiments are saved
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

    paths_experiment_selected = selection_of_experiment(user_selection)

    path_model_1 = ("\\".join(paths_experiment_selected[0].split("\\")[:-1]))
    path_model_2 = ("\\".join(paths_experiment_selected[1].split("\\")[:-1]))

    with open(f"{path_model_1}/test.json") as f:
        results_1 = json.load(f)
    with open(f"{path_model_1}/config.json") as f:
        config_1 = json.load(f)

    with open(f"{path_model_2}/test.json") as f:
        results_2 = json.load(f)
    with open(f"{path_model_2}/config.json") as f:
        config_2 = json.load(f)

    mapping_sensor_with_nodes_model_1 = {}
    for node in results_1.keys():
        mapping_sensor_with_nodes_model_1[config_1["nodes_to_filter"][int(node)]] = node

    mapping_sensor_with_nodes_model_2 = {}
    for node in results_2.keys():
        mapping_sensor_with_nodes_model_2[config_2["nodes_to_filter"][int(node)]] = node

    sensor_select = st.selectbox('Choose the sensor_select', mapping_sensor_with_nodes_model_1.keys())

    metrics = list(results_1["0"]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    federated_node_model_1 = []
    if "Federated" in results_1["0"].keys():
        federated_node_model_1.append(results_1[mapping_sensor_with_nodes_model_1[sensor_select]]["Federated"])
        federated_node_model_1 = pd.DataFrame(federated_node_model_1, columns=multiselect_metrics, index=["sensor in Federation"])

    federated_node_model_2 = []
    if "Federated" in results_1["0"].keys():
        federated_node_model_2.append(results_2[mapping_sensor_with_nodes_model_2[sensor_select]]["Federated"])
        federated_node_model_2 = pd.DataFrame(federated_node_model_2, columns=multiselect_metrics, index=["sensor in Federation"])

    _, c2_title_df, _ = st.columns((2, 1, 2))

    with c2_title_df:
        st.header("sensor in Federation")

    c1_model_1, c2_model_2 = st.columns(2)
    with c1_model_1:
        model_1_name = paths_experiment_selected[0].split("\\")[1]
        st.subheader(f"{model_1_name}")
        st.table(federated_node_model_1.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]).format("{:.2f}"))
    with c2_model_2:
        model_2_name = paths_experiment_selected[1].split("\\")[1]
        st.subheader(f"{model_2_name}")
        st.table(federated_node_model_2.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]).format("{:.2f}"))

    params_model_1 = Params(f'{path_model_1}/config.json')
    params_model_2 = Params(f'{path_model_2}/config.json')
    if (path.exists(f'{path_model_1}/y_pred_fed_0.npy') and
        path.exists(f"{path_model_2}/y_pred_fed_0.npy")):
        plot_slider([path_model_1, path_model_2])
