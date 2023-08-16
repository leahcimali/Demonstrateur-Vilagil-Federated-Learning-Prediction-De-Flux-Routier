###############################################################################
# Libraries
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
from src.Demonstrator_app.utils_streamlit_app import load_experiment_config

from utils_streamlit_app import selection_of_experiment, style_dataframe, load_experiment_results, results_to_stats_dataframe, get_colors_for_results, get_results_for_key, get_name_version_normalized
from StreamData import StreamData

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
    config = load_experiment_config(path_experiment_selected)
    stream_data = StreamData(results, config, path_experiment_selected)
    st.subheader(f"A comparison between federated and local version | Average on {stream_data.size} sensors")
    st.subheader("_It's a general statistic including all the sensors in the calculation_")

    normalized = st.radio("Normalized data ?", ["Yes", "No"], index=1)

    results_sensor_federated = stream_data.get_results_federated(normalized == "Yes")

    st.markdown(f"There is/are :red[**{stream_data.get_nb_sensor_better_in_federation('RMSE')} sensors**] on {len(results_sensor_federated)} improved by the federation")

    stream_data.show_results(normalized=normalized == "Yes")


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
