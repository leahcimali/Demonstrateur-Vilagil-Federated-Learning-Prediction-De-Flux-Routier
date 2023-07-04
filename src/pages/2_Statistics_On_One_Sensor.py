###############################################################################
# Libraries
###############################################################################
import streamlit as st
import pandas as pd


from utils_streamlit_app import get_color_fed_vs_local, selection_of_experiment, style_dataframe, load_experiment_results, load_experiment_config
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
# function(s)
#######################################################################
def render_results(df_fed, df_local):
    color_fed = []
    color_local = []
    for i in range(len(METRICS)):
        if (i < len(METRICS) - 1):  # because "Superior Pred %" metric needs to be superior=True
            col_fed, col_local = get_color_fed_vs_local(df_fed.iloc[i]["Value"], df_local.iloc[i]["Value"], superior=False)
        else:
            col_fed, col_local = get_color_fed_vs_local(df_fed.iloc[i]["Value"], df_local.iloc[i]["Value"], superior=True)
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
        st.table(stats_sensor_local.style.set_table_styles(style_dataframe(stats_sensor_local, colors=color_local, column_index=2)))


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
st.divider()

path_experiment_selected = selection_of_experiment()
if path_experiment_selected is not None:
    results = load_experiment_results(path_experiment_selected)
    config = load_experiment_config(path_experiment_selected)

    def format_selectbox_sensor(value):
        return config["nodes_to_filter"][int(value)]

    sensor_selected = st.sidebar.selectbox('Choose the sensor', results.keys(), format_func=format_selectbox_sensor)

    results_sensor_federated = pd.DataFrame(results[sensor_selected]["Federated"], columns=METRICS, index=["Value"])
    stats_sensor_federated = results_sensor_federated.T.applymap(lambda x: '{:.2f}'.format(x))

    results_sensor_local = pd.DataFrame(results[sensor_selected]["local_only"], columns=METRICS, index=["Value"])
    stats_sensor_local = results_sensor_local.T.applymap(lambda x: '{:.2f}'.format(x))

    render_results(stats_sensor_federated, stats_sensor_local)

    PAGES[page_selectioned](path_experiment_selected, sensor_selected)

else:
    st.header(":red[You don't have experiments to see. (check docs/how_to_visualize_results.md)]")