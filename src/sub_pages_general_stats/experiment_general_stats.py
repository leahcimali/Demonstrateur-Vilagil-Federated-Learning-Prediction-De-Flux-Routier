###############################################################################
# Libraries
###############################################################################
import json
import streamlit as st
import pandas as pd


from utils_streamlit_app import get_color_fed_vs_local, selection_of_experiment, style_dataframe


#######################################################################
# Main
#######################################################################
def experiment_general_stats():
    st.subheader("One Experiment")
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

        nodes = results.keys()  # e.g. keys = ['0', '1', '2', ...]
        mapping_sensor_with_node = {}
        for node in nodes:
            mapping_sensor_with_node[config["nodes_to_filter"][int(node)]] = node  # e.g. nodes_to_filter = [118, 261, 10, ...]

        results_sensor_federated = []
        results_sensor_local = []
        for sensor in mapping_sensor_with_node.keys():  # e.g. keys = [118, 261, 10, ...]
            if "Federated" in results[mapping_sensor_with_node[sensor]].keys():  # e.g. keys = ['Federated', 'local_only']
                federated_node = results[mapping_sensor_with_node[sensor]]["Federated"]
                results_sensor_federated.append(federated_node)
            if "local_only" in results[mapping_sensor_with_node[sensor]].keys():  # e.g. keys = ['Federated', 'local_only']
                local_node = results[mapping_sensor_with_node[sensor]]["local_only"]
                results_sensor_local.append(local_node)

        metrics = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]

        st.subheader(f"A comparison between federated and local version | Average on {len(nodes)} sensors")
        st.subheader("_It's a general statistic including all the sensors in the calculation_")
        if results_sensor_federated != [] and results_sensor_local != []:
            df_federated_node = pd.DataFrame(results_sensor_federated, columns=metrics)
            stats_fed_ver = df_federated_node.describe().T
            stats_fed_ver.drop(columns={'count'}, inplace=True)
            stats_fed_ver = stats_fed_ver.applymap(lambda x: '{:.2f}'.format(x))

            df_local_node = pd.DataFrame(results_sensor_local, columns=metrics)
            stats_local_ver = df_local_node.describe().T
            stats_local_ver.drop(columns={'count'}, inplace=True)
            stats_local_ver = stats_local_ver.applymap(lambda x: '{:.2f}'.format(x))

            color_fed = []
            color_local = []
            for i in range(len(metrics)):
                if (i < 3):  # because "Superior Pred %" metric needs to be superior=True
                    col_fed, col_local = get_color_fed_vs_local(stats_fed_ver.iloc[i]["mean"], stats_local_ver.iloc[i]["mean"], superior=False)
                else:
                    col_fed, col_local = get_color_fed_vs_local(stats_fed_ver.iloc[i]["mean"], stats_local_ver.iloc[i]["mean"], superior=True)
                color_fed.append(col_fed)
                color_local.append(col_local)

            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.subheader("Federated")
                # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
                st.table(stats_fed_ver.style.set_table_styles(style_dataframe(stats_fed_ver, colors=color_fed, column_index=2)))
            with c2:
                st.subheader("Local")
                # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
                st.table(stats_local_ver.style.set_table_styles(style_dataframe(stats_local_ver, colors=color_local, column_index=2)))
