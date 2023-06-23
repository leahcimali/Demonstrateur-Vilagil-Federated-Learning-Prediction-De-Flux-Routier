###############################################################################
# Libraries
###############################################################################
import json
import streamlit as st
import pandas as pd


from utils_streamlit_app import selection_of_experiment, style_dataframe


#######################################################################
# Main
#######################################################################
def experiment_general_stats():
    st.subheader("On one Experiment")
    st.write("""
            * On this page select one experiment to see his results.
                * In the table, you will find the general statistics for both the Local version and\\
                the Federated version on differents metrics.
            """)
    st.divider()
    path_experiment_selected = selection_of_experiment()
    if (path_experiment_selected is not None):
        with open(f"{path_experiment_selected}/test.json") as f:
            results = json.load(f)
        with open(f"{path_experiment_selected}/config.json") as f:
            config = json.load(f)

        nodes = results.keys()
        mapping_sensor_and_node = {}
        for node in nodes:
            mapping_sensor_and_node[config["nodes_to_filter"][int(node)]] = node

        df_local_node = []
        df_federated_node = []
        for sensor in mapping_sensor_and_node.keys():
            if "Federated" in results[mapping_sensor_and_node[sensor]].keys():
                federated_node = results[mapping_sensor_and_node[sensor]]["Federated"]
                df_federated_node.append(federated_node)
            if "local_only" in results[mapping_sensor_and_node[sensor]].keys():
                local_node = results[mapping_sensor_and_node[sensor]]["local_only"]
                df_local_node.append(local_node)

        multiselect_metrics = ["RMSE", "MAE", "SMAPE", "Superior Pred %"]

        st.subheader(f"A comparison between federated and local version on {len(nodes)} sensors")
        st.subheader("_It's a general statistic including all the sensors in the calculation_")
        if df_federated_node != []:
            df_federated_node = pd.DataFrame(df_federated_node, columns=multiselect_metrics)
            global_stats_fed_ver = df_federated_node.describe().T
            global_stats_fed_ver.drop(columns={'count'}, inplace=True)
            global_stats_fed_ver = global_stats_fed_ver.applymap(lambda x: '{:.2f}'.format(x))

        if df_local_node != []:
            df_local_node = pd.DataFrame(df_local_node, columns=multiselect_metrics)
            global_stats_local_ver = df_local_node.describe().T
            global_stats_local_ver.drop(columns={'count'}, inplace=True)
            global_stats_local_ver = global_stats_local_ver.applymap(lambda x: '{:.2f}'.format(x))

        # Create multi-level index for merging
        common_indexes = global_stats_local_ver.index.intersection(global_stats_fed_ver.index)
        multi_index = pd.MultiIndex.from_product([common_indexes, ['Local', 'Federated']], names=['Index', 'Version'])

        merged_stats = pd.concat([global_stats_local_ver, global_stats_fed_ver], axis=0)

        merged_stats.index = multi_index

        st.table(merged_stats.style.set_table_styles(style_dataframe(merged_stats)))
