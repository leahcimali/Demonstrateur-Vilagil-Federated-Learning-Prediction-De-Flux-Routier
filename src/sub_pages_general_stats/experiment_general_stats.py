###############################################################################
# Libraries
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np

from utils_streamlit_app import selection_of_experiment, style_dataframe, load_experiment_results, results_to_stats_dataframe, get_colors_for_results, get_results_for_key, get_name_version_normalized


#######################################################################
# function(s)
#######################################################################
def render_results(df_fed, df_local):
    color_fed, color_local = get_colors_for_results(df_fed, df_local, "mean")

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown("**Federated**")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(df_fed.style.set_table_styles(style_dataframe(df_fed, colors=color_fed, column_index=2)))
    with c2:
        st.markdown("**Local**")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(df_local.style.set_table_styles(style_dataframe(df_local, colors=color_local, column_index=2)))


def render_experiment(path_experiment_selected):
    st.write(path_experiment_selected)
    results = load_experiment_results(path_experiment_selected)
    sensor_index = results.keys()  # e.g. keys = ['0', '1', '2', ...]
    st.subheader(f"A comparison between federated and local version | Average on {len(sensor_index)} sensors")
    st.subheader("_It's a general statistic including all the sensors in the calculation_")

    normalized = st.radio("Normalized data ?", ["Yes", "No"], index=1)
    federated_ver, local_only_ver = get_name_version_normalized(True if normalized == "Yes" else False)

    results_sensor_federated = get_results_for_key(results, sensor_index, federated_ver)
    results_sensor_local = get_results_for_key(results, sensor_index, local_only_ver)
    nb_sensor_better_in_federation = 0
    for i in range(len(results_sensor_federated)):
        if (results_sensor_federated[i]["RMSE"] <= results_sensor_local[i]["RMSE"]):
            nb_sensor_better_in_federation += 1

    st.markdown(f"There is :red[**{nb_sensor_better_in_federation} sensors**] on {len(results_sensor_federated)} improved by the federation")

    metrics = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]
    avg_rate_change = {}
    for metric in metrics:
        for i in range(len(results_sensor_federated)):
            if metric not in avg_rate_change.keys():
                avg_rate_change[metric] = 1 + ((results_sensor_federated[i][metric] - results_sensor_local[i][metric]) / results_sensor_local[i][metric])
            else:
                avg_rate_change[metric] = avg_rate_change[metric] * (1 + ((results_sensor_federated[i][metric] - results_sensor_local[i][metric]) / results_sensor_local[i][metric]))
    for metric in metrics:
        avg_rate_change[metric] = (np.power(avg_rate_change[metric], (1/len(results_sensor_federated))) - 1) * 100
    if results_sensor_federated and results_sensor_local:
        stats_fed_ver = results_to_stats_dataframe(results_sensor_federated)
        stats_local_ver = results_to_stats_dataframe(results_sensor_local)
    mean_stats_fed_ver = stats_fed_ver["mean"]
    mean_stats_local_ver = stats_local_ver["mean"]

    df_mean_diff = pd.DataFrame({"local": mean_stats_local_ver, "diff_on_mean": mean_stats_fed_ver})
    df_mean_diff["diff_on_mean"] = df_mean_diff["diff_on_mean"].astype(float)
    df_mean_diff["local"] = df_mean_diff["local"].astype(float)

    render_results(stats_fed_ver, stats_local_ver)
    st.subheader("Difference between federated and local value on the mean")
    df_mean_diff = df_mean_diff.diff(axis=1)
    df_mean_diff.drop("local", axis=1, inplace=True)
    df_mean_diff = df_mean_diff.applymap(lambda x: '{:.2f}'.format(x))
    st.table(df_mean_diff.style.set_table_styles(style_dataframe(df_mean_diff, colors="#000000", column_index=2)))

    avg_rate_change = pd.DataFrame.from_dict(avg_rate_change, orient="index", columns=["Average rate of change"])
    avg_rate_change = avg_rate_change.applymap(lambda x: '{:.2f} %'.format(x))
    st.table(avg_rate_change.style.set_table_styles(style_dataframe(avg_rate_change, colors="#000000", column_index=2)))

#######################################################################
# Main
#######################################################################
def one_experiment():
    st.subheader("One Experiment")
    st.write("""
        * On this page, select one experiment to see its results.
            * In the table, you will find the general statistics for both the Local version and
            the Federated version on different metrics.
        """)
    st.divider()

    path_experiment_selected = selection_of_experiment()

    if path_experiment_selected is not None:
        render_experiment(path_experiment_selected)
