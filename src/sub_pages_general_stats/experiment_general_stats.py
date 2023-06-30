###############################################################################
# Libraries
###############################################################################
import streamlit as st
import pandas as pd


from utils_streamlit_app import get_color_fed_vs_local, selection_of_experiment, style_dataframe, load_experiment_results


#######################################################################
# Constant(s)
#######################################################################
METRICS = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]


#######################################################################
# function(s)
#######################################################################
def results_to_dataframe(results):
    df = pd.DataFrame(results, columns=METRICS)
    df = df.describe().T
    df.drop(columns={'count'}, inplace=True)
    df = df.applymap(lambda x: '{:.2f}'.format(x))
    return df


def render_results(df_fed, df_local):
    color_fed = []
    color_local = []
    for i in range(len(METRICS)):
        if (i < len(METRICS) - 1):  # because "Superior Pred %" metric needs to be superior=True
            col_fed, col_local = get_color_fed_vs_local(df_fed.iloc[i]["mean"], df_local.iloc[i]["mean"], superior=False)
        else:
            col_fed, col_local = get_color_fed_vs_local(df_fed.iloc[i]["mean"], df_local.iloc[i]["mean"], superior=True)
        color_fed.append(col_fed)
        color_local.append(col_local)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.subheader("Federated")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(df_fed.style.set_table_styles(style_dataframe(df_fed, colors=color_fed, column_index=2)))
    with c2:
        st.subheader("Local")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(df_local.style.set_table_styles(style_dataframe(df_local, colors=color_local, column_index=2)))


def get_results_for_key(results, sensors, key):
    return [
        results[sensor][key]
        for sensor in sensors
        if key in results[sensor].keys()
    ]


def render_experiment(path_experiment_selected):
    results = load_experiment_results(path_experiment_selected)

    nodes = results.keys()  # e.g. keys = ['0', '1', '2', ...]

    results_sensor_federated = get_results_for_key(results, nodes, "Federated")
    results_sensor_local = get_results_for_key(results, nodes, "local_only")

    st.subheader(f"A comparison between federated and local version | Average on {len(nodes)} sensors")
    st.subheader("_It's a general statistic including all the sensors in the calculation_")
    if results_sensor_federated and results_sensor_local:
        stats_fed_ver = results_to_dataframe(results_sensor_federated)
        stats_local_ver = results_to_dataframe(results_sensor_local)

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
