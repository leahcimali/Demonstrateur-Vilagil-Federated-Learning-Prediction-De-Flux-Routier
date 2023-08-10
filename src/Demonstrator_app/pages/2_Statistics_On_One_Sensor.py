###############################################################################
# Libraries
###############################################################################
import streamlit as st


from utils_streamlit_app import results_to_dataframe, get_colors_for_results, selection_of_experiment, style_dataframe, load_experiment_results, load_experiment_config, get_results_for_key, get_name_version_normalized
from sub_pages_one_sensor.box_plot import box_plot_sensor
from sub_pages_one_sensor.predictions_graph import prediction_graph_sensor
from sub_pages_one_sensor.single_sensor_map import single_sensor_map_sensor

import pandas as pd
import numpy as np


#######################################################################
# Constant(s)
#######################################################################
PAGES = {
    "Prediction sensor": prediction_graph_sensor,
    "Boxplot sensor": box_plot_sensor,
    "Map of sensor": single_sensor_map_sensor
}


#######################################################################
# function(s)
#######################################################################
def render_results(df_fed, df_local):
    color_fed, color_local = get_colors_for_results(df_fed, df_local, "Value")

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.subheader("Federated")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(df_fed.style.set_table_styles(style_dataframe(df_fed, colors=color_fed, column_index=2)))
    with c2:
        st.subheader("Local")
        # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
        st.table(df_local.style.set_table_styles(style_dataframe(df_local, colors=color_local, column_index=2)))


#######################################################################
# Main
#######################################################################
st.header("Statistics on one sensor")
st.divider()
st.markdown("""
            This is the statistics page for a single sensor. On the sidebar,\\
            you can see different visualizations.
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
st.divider()

path_experiment_selected = selection_of_experiment()
if path_experiment_selected is not None:
    results = load_experiment_results(path_experiment_selected)
    config = load_experiment_config(path_experiment_selected)

    def format_selectbox_sensor(value):
        return config["nodes_to_filter"][int(value)]

    sensor_selected = st.sidebar.selectbox('Choose the sensor', results.keys(), format_func=format_selectbox_sensor)

    stats_sensor_federated = results_to_dataframe(results, sensor_selected, "Federated_unormalized")
    stats_sensor_local = results_to_dataframe(results, sensor_selected, "local_only_unormalized")

    st.subheader(f"Working on sensor {config['nodes_to_filter'][int(sensor_selected)]}")

    render_results(stats_sensor_federated, stats_sensor_local)

    federated_ver, local_only_ver = get_name_version_normalized(False)

    results_sensor_federated = get_results_for_key(results, sensor_selected, federated_ver)
    results_sensor_local = get_results_for_key(results, sensor_selected, local_only_ver)

    metrics = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]
    avg_rate_change = {}
    for metric in metrics:
        for i in range(len(results_sensor_federated)):
            if metric not in avg_rate_change.keys():
                avg_rate_change[metric] = 1 + ((results_sensor_federated[i][metric] - results_sensor_local[i][metric]) / results_sensor_local[i][metric])
            else:
                avg_rate_change[metric] = avg_rate_change[metric] * (1 + ((results_sensor_federated[i][metric] - results_sensor_local[i][metric]) / results_sensor_local[i][metric]))
    for metric in metrics:
        avg_rate_change[metric] = (np.power(avg_rate_change[metric], (1 / len(results_sensor_federated))) - 1) * 100

    avg_rate_change = pd.DataFrame.from_dict(avg_rate_change, orient="index", columns=["Average rate of change"])
    avg_rate_change = avg_rate_change.applymap(lambda x: '{:.2f} %'.format(x))
    st.table(avg_rate_change.style.set_table_styles(style_dataframe(avg_rate_change, colors="#000000", column_index=2)))

    PAGES[page_selectioned](path_experiment_selected, sensor_selected)

else:
    st.header(":red[You don't have experiments to see. (check docs/how_to_visualize_results.md)]")
