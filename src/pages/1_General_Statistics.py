###############################################################################
# Libraries
###############################################################################
import streamlit as st


from sub_pages_general_stats.experiment_general_stats import one_experiment
from sub_pages_general_stats.comparison_experiments import comparison_experiments
from sub_pages_general_stats.map_of_sensors import map_of_sensors


st.set_page_config(layout="wide")


#######################################################################
# Constant(s)
#######################################################################
PAGES = {
    "Map of sensors": map_of_sensors,
    "One experiment": one_experiment,
    "Comparison between experiments": comparison_experiments
}


#######################################################################
# Main
#######################################################################
st.header("General Statistics")
st.divider()
st.markdown("""
            The general statistics are calculated as the mean of all results
            obtained from each sensor. \\
            This involves calculating the average of the residuals predictions
            obtained from each individual sensor with different metrics like
            RMSE, MSE, ... \\
            Then aggregating them to obtain the overall mean.\\
            This approach provides a comprehensive measure that
            represents the collective data from all sensors.
            """)
st.divider()

st.sidebar.title("General Statistics")
with st.sidebar:
    page_selectioned = st.radio(
        "Choose what you want to see",
        PAGES.keys(),
        index=0
    )

PAGES[page_selectioned]()
