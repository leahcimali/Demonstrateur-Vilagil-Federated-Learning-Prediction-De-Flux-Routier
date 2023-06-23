###############################################################################
# Libraries
###############################################################################
import streamlit as st


from sub_pages_general_stats.experiment_general_stats import experiment_general_stats
from sub_pages_general_stats.comparison_models import comparison_models


st.set_page_config(layout="wide")


#######################################################################
# Constant(s)
#######################################################################
PAGES = {
    "One experiment": experiment_general_stats,
    "Comparison between models": comparison_models
}


#######################################################################
# Main
#######################################################################
st.header("General Statistics")
st.markdown("---")
st.markdown("""
            The general statistics are calculated as the mean of all results obtained from each sensor. This involves calculating\\
            the average of the values obtained from each individual sensor and then aggregating them to obtain the overall mean.\\
            This approach provides a comprehensive measure that represents the collective data from all sensors.
            """)
st.markdown("---")

st.sidebar.title("General Statistics")
with st.sidebar:
    page_selectioned = st.radio(
        "Choose what you want to see",
        PAGES.keys(),
        index=0
    )

PAGES[page_selectioned]()
