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


from metrics import rmse
from config import Params
from utils_streamlit_app import load_numpy, map_path_experiments_to_params, selection_of_experiment
from utils_streamlit_app import get_color_fed_vs_local

st.set_page_config(layout="wide")


def plot_prediction_graph(experiment_path):
    test_set = load_numpy(f"{experiment_path}/test_data_{mapping_sensor_with_nodes[sensor_select]}.npy")

    y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred = load_numpy(f"{experiment_path}/y_pred_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")

    index = load_numpy(f"{experiment_path}/index_{mapping_sensor_with_nodes[sensor_select]}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')

    slider = st.slider('Select time?', 0, len(index) - params.prediction_horizon - params.window_size, params.prediction_horizon)

    def plot_graph(color, label, title, y_pred, rmse_value, i):
        df = pd.DataFrame({'Time': index[i:i + params.window_size + params.prediction_horizon], 'Traffic Flow': test_set[i:i + params.window_size + params.prediction_horizon].flatten()})
        df['Window'] = df['Traffic Flow'].where((df['Time'] >= index[i]) & (df['Time'] <= index[i + params.window_size - 1]))
        df['y_true'] = df['Traffic Flow'].where(df['Time'] >= index[i + params.window_size - 1])
        df[f'y_pred_{label}'] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred[i, :]])
        fig = px.line(df, x='Time', y=["Window"], color_discrete_sequence=['black'])
        fig.add_scatter(x=df['Time'], y=df['y_true'], mode='lines', marker=dict(color='blue'), name='y_true')
        fig.add_scatter(x=df['Time'], y=df[f'y_pred_{label}'], mode='markers+lines', marker=dict(color=color), name=f'{label} RMSE : {rmse_value:.2f}')
        fig.add_bar(x=df['Time'], y=(np.abs(df[f'y_pred_{label}'] - df['y_true'])), name='Absolute Error')
        fig.update_xaxes(tickformat='%H:%M', dtick=3600000)
        fig.update_layout(xaxis_title='Time', yaxis_title='Traffic Flow', title=f"| {title} | {index[slider+params.window_size].strftime(f'Day: %Y-%m-%d | Time prediction: {int(params.prediction_horizon*5/60)}h (%Hh-%Mmin')} to {index[slider + params.window_size + params.prediction_horizon].strftime('%Hh-%Mmin) |')} ", title_font=dict(size=28), legend=dict(title='Legends', font=dict(size=16)))
        fig.add_vrect(x0=index[i], x1=index[i + params.window_size - 1], fillcolor='gray', opacity=0.2, line_width=0)
        return fig

    rmse_local = rmse(y_true[slider, :].flatten(), y_pred[slider, :].flatten())
    rmse_fed = rmse(y_true[slider, :].flatten(), y_pred_fed[slider, :].flatten())

    color_fed, color_local = get_color_fed_vs_local(rmse_fed, rmse_local, superior=False)

    # FEDERATED
    fed_fig = plot_graph(color_fed, 'Federated', "Federated Prediction", y_pred_fed, rmse_fed, slider)

    # LOCAL
    local_fig = plot_graph(color_local, 'Local', "Local Prediction", y_pred, rmse_local, slider)

    with st.spinner('Plotting...'):
        st.plotly_chart(fed_fig, use_container_width=True)
        st.plotly_chart(local_fig, use_container_width=True)


#######################################################################
# Main
#######################################################################
st.header("Predictions Graph")

experiments = "./experiments/"  # PATH where your experiments are saved
if path_files := glob.glob(f"./{experiments}**/config.json", recursive=True):

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
    sensor_select = st.selectbox('Choose the sensor', mapping_sensor_with_nodes.keys(), index=st.session_state['sensor_select'])
    st.session_state['sensor_select'] = int(mapping_sensor_with_nodes[sensor_select])

    metrics = list(results[mapping_sensor_with_nodes[sensor_select]]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    local_node = []
    if "local_only" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        local_node = results[mapping_sensor_with_nodes[sensor_select]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=multiselect_metrics, index=["sensor alone"])

    federated_node = []
    if "Federated" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        federated_node = results[mapping_sensor_with_nodes[sensor_select]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=multiselect_metrics, index=["sensor in Federation"])

    st.subheader("sensor in Federation vs sensor alone")
    fed_local_node = pd.concat((federated_node, local_node), axis=0)
    st.table(fed_local_node.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]).format("{:.2f}"))

    params = Params(f'{path_experiment_selected}/config.json')
    if (path.exists(f'{path_experiment_selected}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy') and
        path.exists(f"{path_experiment_selected}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")):
        plot_prediction_graph(path_experiment_selected)
    else:
        st.write("ERROR")
