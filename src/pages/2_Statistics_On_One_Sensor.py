###############################################################################
# Libraries
###############################################################################
import json


import streamlit as st
import pandas as pd


from utils_streamlit_app import get_color_fed_vs_local, selection_of_experiment, style_dataframe
from sub_pages_one_sensor.box_plot import box_plot_sensor
from sub_pages_one_sensor.predictions_graph import prediction_graph_sensor
from sub_pages_one_sensor.single_sensor_map import single_sensor_map_sensor


st.set_page_config(layout="wide")


#######################################################################
# Constant(s)
#######################################################################
PAGES = {
    "Prediction sensor": prediction_graph_sensor,
    "Boxplot sensor": box_plot_sensor,
    "Map of sensor": single_sensor_map_sensor
}

METRICS = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]


#######################################################################
# Main
#######################################################################
st.header("Statistics on one sensor")
st.divider()
st.markdown("""
            This is the statistics page for a single sensor. On the sidebar,\\
            you can view different visualizations.
            """)
st.divider()

st.sidebar.title("Statistics on one sensor")
with st.sidebar:
    page_selectioned = st.radio(
        "Choose the visualization",
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

    sensor_selected = st.sidebar.selectbox('Choose the sensor', mapping_sensor_with_node.keys())

    results_sensor_federated = []
    if "Federated" in results[mapping_sensor_with_node[sensor_selected]].keys():
        results_sensor_federated = pd.DataFrame(results[mapping_sensor_with_node[sensor_selected]]["Federated"], columns=METRICS, index=["Value"])
        stats_sensor_federated = results_sensor_federated.T

    results_sensor_local = []
    if "local_only" in results[mapping_sensor_with_node[sensor_selected]].keys():
        results_sensor_local = pd.DataFrame(results[mapping_sensor_with_node[sensor_selected]]["local_only"], columns=METRICS, index=["Value"])
        stats_sensor_local = results_sensor_local.T

    color_fed = []
    color_local = []
    for i in range(len(METRICS)):
        if (i < 3):  # because "Superior Pred %" metric needs to be superior=True
            col_fed, col_local = get_color_fed_vs_local(stats_sensor_federated.iloc[i]["Value"], stats_sensor_local.iloc[i]["Value"], superior=False)
        else:
            col_fed, col_local = get_color_fed_vs_local(stats_sensor_federated.iloc[i]["Value"], stats_sensor_local.iloc[i]["Value"], superior=True)
        color_fed.append(col_fed)
        color_local.append(col_local)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.subheader("Federated")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(stats_sensor_federated.style.set_table_styles(style_dataframe(stats_sensor_federated, colors=color_fed, column_index=2)))
    with c2:
        st.subheader("Local")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(stats_sensor_local.style.set_table_styles(style_dataframe(stats_sensor_local, colors=color_local, column_index=2)))

    PAGES[page_selectioned](path_experiment_selected, mapping_sensor_with_node[sensor_selected])
