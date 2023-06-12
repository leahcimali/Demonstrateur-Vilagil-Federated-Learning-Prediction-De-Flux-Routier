###############################################################################
# Libraries
###############################################################################
import glob

import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils_streamlit_app import map_path_experiments_to_params, selection_of_experiment

st.set_page_config(layout="wide")

#######################################################################
# Main
#######################################################################
st.header("General Statistics")

experiments = "experiments"
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

    nodes = results.keys()
    mapping_sensor_and_node = {}
    for node in results.keys():
        mapping_sensor_and_node[config["nodes_to_filter"][int(node)]] = node

    df_federated_node = []
    for sensor in mapping_sensor_and_node.keys():
        if "Federated" in results[mapping_sensor_and_node[sensor]].keys():
            federated_node = results[mapping_sensor_and_node[sensor]]["Federated"]
            df_federated_node.append(federated_node)

    if df_federated_node != []:
        st.subheader("How much the Federated version is performant on average taking in account all the sensors for the calculation of the statistics")
        df_federated_node = pd.DataFrame(df_federated_node)
        global_stats_fed_ver = df_federated_node.describe().T
        global_stats_fed_ver.rename(columns={'count': 'Nb sensors'}, inplace=True)
        st.table(global_stats_fed_ver.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]))

        c1_boxplot_fed, c2_bobxplot_fed, c3_boxplot_fed = st.columns((1, 2, 1))
        fig, ax = plt.subplots()
        bar_plot_results = df_federated_node
        bar_plot_results.boxplot(column=["RMSE", "MAE"], ylabel="values", xlabel="sensor", ax=ax)
        plt.yticks(np.arange(0, max(bar_plot_results["RMSE"].max(), bar_plot_results["MAE"].max()), 10))
        with c2_bobxplot_fed:
            st.pyplot(fig, use_container_width=True)
