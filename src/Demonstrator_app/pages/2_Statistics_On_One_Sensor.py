###############################################################################
# Libraries
###############################################################################
import streamlit as st


from utils_streamlit_app import get_colors_for_results, selection_of_experiment, style_dataframe, load_experiment_results, load_experiment_config
from sub_pages_one_sensor.box_plot import box_plot_sensor
from sub_pages_one_sensor.predictions_graph import prediction_graph_sensor
from sub_pages_one_sensor.single_sensor_map import single_sensor_map_sensor
from StreamData import StreamData

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
    stream_data = StreamData(results, config)

    def format_selectbox_sensor(value):
        return stream_data.parameters["nodes_to_filter"][int(value)]

    sensor_selected = st.sidebar.selectbox('Choose the sensor', stream_data.indexes, format_func=format_selectbox_sensor)

    st.subheader(f"Working on sensor {stream_data.sensors_name[int(sensor_selected)]}")
    stream_data.show_results_sensor(sensor_selected, False)

    PAGES[page_selectioned](path_experiment_selected, sensor_selected)

else:
    st.header(":red[You don't have experiments to see. (check docs/how_to_visualize_results.md)]")
