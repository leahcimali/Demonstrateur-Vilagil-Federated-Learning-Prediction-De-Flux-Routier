###############################################################################
# Libraries
###############################################################################
import streamlit as st
import pandas as pd


from utils_streamlit_app import selection_of_experiment, style_dataframe, load_experiment_results, results_to_stats_dataframe, get_colors_for_results, get_results_for_key


#######################################################################
# function(s)
#######################################################################
def render_results(df_fed, df_local):
    color_fed, color_local = get_colors_for_results(df_fed, df_local, "mean")

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.subheader("Federated")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(df_fed.style.set_table_styles(style_dataframe(df_fed, colors=color_fed, column_index=2)))
    with c2:
        st.subheader("Local")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(df_local.style.set_table_styles(style_dataframe(df_local, colors=color_local, column_index=2)))


def render_experiment(path_experiment_selected):
    results = load_experiment_results(path_experiment_selected)

    sensor_index = results.keys()  # e.g. keys = ['0', '1', '2', ...]

    results_sensor_federated = get_results_for_key(results, sensor_index, "Federated")
    results_sensor_local = get_results_for_key(results, sensor_index, "local_only")

    st.subheader(f"A comparison between federated and local version | Average on {len(sensor_index)} sensors")
    st.subheader("_It's a general statistic including all the sensors in the calculation_")
    if results_sensor_federated and results_sensor_local:
        stats_fed_ver = results_to_stats_dataframe(results_sensor_federated)
        stats_local_ver = results_to_stats_dataframe(results_sensor_local)

    render_results(stats_fed_ver, stats_local_ver)


#######################################################################
# Main
#######################################################################
def experiment_general_stats():
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
