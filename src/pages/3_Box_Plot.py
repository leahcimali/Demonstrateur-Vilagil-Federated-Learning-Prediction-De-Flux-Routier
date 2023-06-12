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


from utils_streamlit_app import load_numpy, map_path_experiments_to_params, selection_of_experiment
from config import Params

st.set_page_config(layout="wide")


def plot_slider(experiment_path):
    y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred = load_numpy(f"{experiment_path}/y_pred_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")

    def plot_box(title, ae, max_y_value, color):
        fig = px.box(y=ae, color_discrete_sequence=[color], title=title, points="outliers")
        fig.update_layout(
            title={
                'text': f"{title}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title=f"sensor_select {sensor_select}",
            yaxis_title="Trafic flow (absolute error)",
            yaxis=dict(range=[0, max_y_value]),
            font=dict(
                size=28,
                color="#7f7f7f"
            ), height=900, width=250
        )
        return fig

    ae_fed = (np.abs(y_pred_fed.flatten() - y_true.flatten()))
    ae_local = (np.abs(y_pred.flatten() - y_true.flatten()))

    max_y_value = max(max(ae_fed), max(ae_local))

    # FEDERATED
    fed_fig = plot_box("Federated Prediction", ae_fed, max_y_value, 'green')

    # LOCAL
    local_fig = plot_box("Alone Prediction", ae_local, max_y_value, 'red')

    with st.spinner('Plotting...'):
        st.subheader(f"Comparison between Federation and local version on sensor {sensor_select} (Absolute Error)")
        _, c2_fed_fig, c3_local_fig, _ = st.columns((1, 1, 1, 1))
        with c2_fed_fig:
            st.plotly_chart(fed_fig, use_container_width=False)
        with c3_local_fig:
            st.plotly_chart(local_fig, use_container_width=False)


#######################################################################
# Main
#######################################################################


st.header("Box Plot")

experiments = "./experiments/"  # PATH where your experiments are saved
if path_files := glob.glob(f"{experiments}/**/config.json", recursive=True):

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

    mapping_sensor_with_nodes = {}
    for node in results.keys():
        mapping_sensor_with_nodes[config["nodes_to_filter"][int(node)]] = node

    if 'sensor_select' not in st.session_state:
        st.session_state['sensor_select'] = 0
    sensor_select = st.selectbox('Choose the sensor_select', mapping_sensor_with_nodes.keys(), index=st.session_state['sensor_select'])
    st.session_state['sensor_select'] = int(mapping_sensor_with_nodes[sensor_select])

    metrics = list(results[mapping_sensor_with_nodes[sensor_select]]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    federated_node = []
    if "Federated" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        federated_node = results[mapping_sensor_with_nodes[sensor_select]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=multiselect_metrics, index=["sensor in Federation"])

    local_node = []
    if "local_only" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        local_node = results[mapping_sensor_with_nodes[sensor_select]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=multiselect_metrics, index=["sensor alone"])

    st.subheader("sensor in Federation vs sensor alone")
    fed_local_node = pd.concat((federated_node, local_node), axis=0)
    st.table(fed_local_node.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]).format("{:.2f}"))

    params = Params(f'{path_experiment_selected}/config.json')
    if (path.exists(f'{path_experiment_selected}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy') and
        path.exists(f"{path_experiment_selected}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")):
        plot_slider(path_experiment_selected)
