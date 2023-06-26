###############################################################################
# Libraries
###############################################################################
import json


import streamlit as st
import pandas as pd


from utils_streamlit_app import selection_of_experiment, style_dataframe
from sub_pages_one_sensor.box_plot import box_plot_sensor
from sub_pages_one_sensor.predictions_graph import prediction_graph_sensor
from sub_pages_one_sensor.map import map_sensors


st.set_page_config(layout="wide")


#######################################################################
# Constant(s)
#######################################################################
PAGES = {
    "Prediction sensor": prediction_graph_sensor,
    "Boxplot sensor": box_plot_sensor,
    "Map of sensors": map_sensors
}


#######################################################################
# Main
#######################################################################
st.header("Statistics on one sensor")
st.divider()
st.markdown("""
            This is the statistics page for a single sensor. Here, you can view metrics
            for the sensor of your choice.
            """)
st.divider()

st.sidebar.title("Statistics on one sensor")
with st.sidebar:
    page_selectioned = st.radio(
        "Choose what you want to see",
        PAGES.keys(),
        index=0
    )

st.subheader("Selection of the experiment")

path_experiment_selected = selection_of_experiment()
if (path_experiment_selected is not None):
    with open(f"{path_experiment_selected}/test.json") as f:
        results = json.load(f)
    with open(f"{path_experiment_selected}/config.json") as f:
        config = json.load(f)

    mapping_sensor_with_node = {}
    for node in results.keys():  # e.g. keys = ['0', '1', '2', ...]
        mapping_sensor_with_node[config["nodes_to_filter"][int(node)]] = node  # e.g. nodes_to_filter = [118, 261, 10, ...]

    st.divider()

    sensor_selected = st.selectbox('Choose the sensor', mapping_sensor_with_node.keys())

    metrics = ["RMSE", "MAE", "SMAPE", "Superior Pred %"]

    results_sensor_federated = []
    if "Federated" in results[mapping_sensor_with_node[sensor_selected]].keys():  # e.g. keys = [118, 261, 10, ...]
        results_sensor_federated = results[mapping_sensor_with_node[sensor_selected]]["Federated"]
        results_sensor_federated = pd.DataFrame(results_sensor_federated, columns=metrics, index=["sensor in Federation"])

    results_sensor_local = []
    if "local_only" in results[mapping_sensor_with_node[sensor_selected]].keys():  # e.g. keys = [118, 261, 10, ...]
        results_sensor_local = results[mapping_sensor_with_node[sensor_selected]]["local_only"]
        results_sensor_local = pd.DataFrame(results_sensor_local, columns=metrics, index=["sensor in Local"])

    st.subheader("sensor in Federation vs sensor in Local")
    results_fed_local = pd.concat((results_sensor_federated, results_sensor_local), axis=0)

    # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
    st.table(results_fed_local.style.set_table_styles(style_dataframe(results_fed_local)).format("{:.2f}"))

    PAGES[page_selectioned](path_experiment_selected, mapping_sensor_with_node[sensor_selected])
