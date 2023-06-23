###############################################################################
# Libraries
###############################################################################
import json


import streamlit as st
import pandas as pd


from utils_streamlit_app import selection_of_experiment, style_dataframe
from sub_page_one_sensor.box_plot import box_plot_sensor
from sub_page_one_sensor.predictions_graph import prediction_graph_sensor
from sub_page_one_sensor.map import map_sensors

st.set_page_config(layout="wide")

#######################################################################
# Constant(s)
#######################################################################
PAGES = {
    "Prediction sensor": prediction_graph_sensor,
    "Box Plot sensor": box_plot_sensor,
    "Map of sensors": map_sensors
}


#######################################################################
# Main
#######################################################################
st.header("Statistics on one sensor")
st.markdown("---")
st.markdown("""
            TODO DESCRIPTION
            """)
st.markdown("---")

st.sidebar.title("Statistics on one sensor")
with st.sidebar:
    page_selectioned = st.radio(
        "Choose what you want to see",
        PAGES.keys(),
        index=0
    )

path_experiment_selected = selection_of_experiment()
if (path_experiment_selected is not None):
    with open(f"{path_experiment_selected}/test.json") as f:
        results = json.load(f)
    with open(f"{path_experiment_selected}/config.json") as f:
        config = json.load(f)

    mapping_sensor_with_nodes = {}
    for node in results.keys():
        mapping_sensor_with_nodes[config["nodes_to_filter"][int(node)]] = node

    sensor_select = st.selectbox('Choose the sensor', mapping_sensor_with_nodes.keys())

    metrics = list(results[mapping_sensor_with_nodes[sensor_select]]["local_only"].keys())
    multiselect_metrics = ["RMSE", "MAE", "SMAPE", "Superior Pred %"]

    local_node = []
    if "local_only" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        local_node = results[mapping_sensor_with_nodes[sensor_select]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=multiselect_metrics, index=["sensor in Local"])

    federated_node = []
    if "Federated" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        federated_node = results[mapping_sensor_with_nodes[sensor_select]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=multiselect_metrics, index=["sensor in Federation"])

    st.subheader("sensor in Federation vs sensor in Local")
    fed_local_node = pd.concat((federated_node, local_node), axis=0)
    st.table(fed_local_node.style.set_table_styles(style_dataframe(fed_local_node)).format("{:.2f}"))

    PAGES[page_selectioned](path_experiment_selected, mapping_sensor_with_nodes[sensor_select])
