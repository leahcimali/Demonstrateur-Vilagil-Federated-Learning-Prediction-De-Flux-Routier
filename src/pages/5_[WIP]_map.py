from os import path


import streamlit as st
from streamlit_folium import folium_static

import glob
import json
import pandas as pd
import numpy as np
import plotly.express as px
import folium


from metrics import rmse, maape
from config import Params
from utils_streamlit_app import load_numpy, map_path_experiments_to_params
from utils_streamlit_app import create_circle_precision_predict
from utils_streamlit_app import selection_of_experiment


st.set_page_config(layout="wide")

seattle_roads = [
    [47.679470, -122.315626],
    [47.679441, -122.306665],
    [47.683058163418266, -122.30074031156877],
    [47.67941986097163, -122.29031294544225],
    [47.67578888921566, -122.30656814568495],
    [47.67575649888934, -122.29026613694701],
    [47.68307457244817, -122.29054200791231],
    [47.68300028244276, -122.3121427044953],
    [47.670728396123444, -122.31192781883172],
    [47.675825, -122.315658],
    [47.69132417321706, -122.31221442807933],
    [47.68645681961068, -122.30076590191602],
    [47.68304467808857, -122.27975989945097],
    [47.6974488132659, -122.29057907732675]
]

###############################################################################
# Map Folium
###############################################################################
# Create maps centered on Seattle
seattle_map_global = folium.Map(location=[47.6776, -122.30064], zoom_start=15, zoomSnap=0.25)
seattle_map_local = folium.Map(location=[47.67763, -122.30064], zoom_start=15, zoomSnap=0.25)

st.title('[WIP] carte Analyses results experimentation')


def plot_prediction_graph(experiment_path):
    test_set = load_numpy(f"{experiment_path}/test_data_{mapping_sensor_with_nodes[sensor_select]}.npy")

    y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred = load_numpy(f"{experiment_path}/y_pred_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")

    index = load_numpy(f"{experiment_path}/index_{mapping_sensor_with_nodes[sensor_select]}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')

    slider = st.slider('Select time?', 0, len(index) - params.prediction_horizon - params.window_size, params.prediction_horizon)

    def plot_graph(color, label, title, y_pred, i):
        rmse_computed = rmse(y_true[i, :].flatten(), y_pred[i, :].flatten())

        df = pd.DataFrame({'Time': index[i:i + params.window_size + params.prediction_horizon], 'Traffic Flow': test_set[i:i + params.window_size + params.prediction_horizon].flatten()})
        df['Window'] = df['Traffic Flow'].where((df['Time'] >= index[i]) & (df['Time'] <= index[i + params.window_size - 1]))
        df['y_true'] = df['Traffic Flow'].where(df['Time'] >= index[i + params.window_size - 1])
        df[f'y_pred_{label}'] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred[i, :]])
        fig = px.line(df, x='Time', y=["Window"], color_discrete_sequence=['black'])
        fig.add_scatter(x=df['Time'], y=df['y_true'], mode='markers+lines', marker=dict(color='blue'), name='y_true')
        fig.add_scatter(x=df['Time'], y=df[f'y_pred_{label}'], mode='markers+lines', marker=dict(color=color), name=f'{label} RMSE : {rmse_computed}')
        fig.add_bar(x=df['Time'], y=(np.abs(df[f'y_pred_{label}'] - df['y_true'])), name='Absolute Error')
        fig.update_xaxes(tickformat='%H:%M', dtick=3600000)
        fig.update_layout(xaxis_title='Time', yaxis_title='Traffic Flow', title=f"{title} ({index[slider].strftime('%Y-%m-%d')})", title_font=dict(size=28), legend=dict(title='Legends', font=dict(size=16)))
        fig.add_vrect(x0=index[i], x1=index[i + params.window_size - 1], fillcolor='gray', opacity=0.2, line_width=0)

        return fig

    # FEDERATED
    fed_fig = plot_graph('green', 'Federated', "Federated Prediction", y_pred_fed, slider)

    # LOCAL
    # local_fig = plot_graph('red', 'Local', "Local Prediction", y_pred, slider)

    with st.spinner('Plotting...'):
        st.plotly_chart(fed_fig, use_container_width=True)
    # st.plotly_chart(local_fig, use_container_width=True)


def plot_map(experiment_path):
    # test_set = load_numpy(f"{experiment_path}/test_data_{mapping_sensor_with_nodes[sensor]}.npy")
    # sensors_results = {}
    # for sensor in mapping_sensor_with_nodes.keys():
    #     if sensor in sensors_results.keys():
    #         sensors_results[f"{sensor}"]["y_true"] = load_numpy(f"{experiment_path}/y_true_local_{mapping_sensor_with_nodes[sensor]}.npy")
    #         sensors_results[f"{sensor}"]["y_pred"] = load_numpy(f"{experiment_path}/y_pred_local_{mapping_sensor_with_nodes[sensor]}.npy")
    #         sensors_results[f"{sensor}"]["y_pred_fed"] = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_sensor_with_nodes[sensor]}.npy")
    #     else:
    #         sensor[f"{sensor}"] = {}

    index = load_numpy(f"{experiment_path}/index_{mapping_sensor_with_nodes[sensor_select]}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')

    slider = st.slider('Select time?', 0, len(index) - params.prediction_horizon - params.window_size, params.prediction_horizon, key="test")

    def plot_map_slider(y_true, y_pred, y_pred_fed, i, coords):
        red = "#fe7597"
        green = "#7cff2d"
        maape_computed_local = 1 - (maape(y_true[i, :].flatten(), y_pred[i, :].flatten())) / 100.0
        maape_computed_fed = 1 - (maape(y_true[i, :].flatten(), y_pred_fed[i, :].flatten())) / 100.0
        if ((maape_computed_local) > (maape_computed_fed)):
            color_local = green
            color_fed = red
        else:
            color_local = red
            color_fed = green

        create_circle_precision_predict(coords, maape_computed_local, seattle_map_local, color_local)
        create_circle_precision_predict(coords, maape_computed_fed, seattle_map_global, color_fed)

    for sensor in mapping_sensor_with_nodes.keys():
        y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_sensor_with_nodes[sensor]}.npy")
        y_pred = load_numpy(f"{experiment_path}/y_pred_local_{mapping_sensor_with_nodes[sensor]}.npy")
        y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_sensor_with_nodes[sensor]}.npy")
        plot_map_slider(y_true, y_pred, y_pred_fed, slider, map_sensor_loc[sensor])

    seattle_map_global.fit_bounds(seattle_map_global.get_bounds(), padding=(10, 10))
    seattle_map_local.fit_bounds(seattle_map_local.get_bounds(), padding=(10, 10))

    # Create a table
    col1, col2 = st.columns(2, gap="small")
    with col1:
        col1.header('Federated model results')
        folium_static(seattle_map_global, width=650)
    with col2:
        col2.header('Local models results')
        folium_static(seattle_map_local, width=650)

#######################################################################
# Main
#######################################################################


st.header("Predictions Graph")

experiments = "./experiments/"  # PATH where your experiments are saved
if path_files := glob.glob(f"{experiments}**/config.json", recursive=True):

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

    map_sensor_loc = {}
    seattle_roads_crop = [seattle_roads[i] for i in range(len(mapping_sensor_with_nodes.keys()))]

    for sensor, locations in zip(mapping_sensor_with_nodes.keys(), seattle_roads_crop):
        map_sensor_loc[sensor] = locations

    for sensor in map_sensor_loc.keys():
        tooltip = f"Road: {sensor}"
        folium.Marker(location=map_sensor_loc[sensor], tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_global)
        folium.Marker(location=map_sensor_loc[sensor], tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_local)

    metrics = list(results[mapping_sensor_with_nodes[sensor_select]]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    local_node = []
    if "local_only" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        local_node = results[mapping_sensor_with_nodes[sensor_select]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=multiselect_metrics, index=["Sensor alone"])
    federated_node = []
    if "Federated" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        federated_node = results[mapping_sensor_with_nodes[sensor_select]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=multiselect_metrics, index=["Sensor in Federation"])

    st.subheader("Sensor in Federation vs Sensor alone")
    fed_local_node = pd.concat((federated_node, local_node), axis=0)
    st.table(fed_local_node.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]).format("{:.2f}"))

    params = Params(f'{path_experiment_selected}/config.json')
    if (path.exists(f'{path_experiment_selected}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy') and
        path.exists(f"{path_experiment_selected}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")):
        plot_prediction_graph(path_experiment_selected)
        plot_map(path_experiment_selected)
