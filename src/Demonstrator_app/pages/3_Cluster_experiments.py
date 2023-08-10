import glob
from pathlib import PurePath
import streamlit as st


from sub_pages_cluster_experiments import One_Cluster, All_Clusters
from utils_streamlit_app import load_experiment_config, load_experiment_results, selection_of_experiment_cluster
from ClusterData import ClusterData


#######################################################################
# Constant(s)
#######################################################################
PAGES = {
    "One Cluster": One_Cluster.one_cluster,
    "All Clusters": All_Clusters.all_clusters,
}

#######################################################################
# Main
#######################################################################
st.header("Cluster experiments")
st.markdown("""
            On the sidebar, you can see different visualizations.
            """)
st.divider()

if experiments_path := glob.glob(f"./{selection_of_experiment_cluster()}/**/config.json", recursive=True):
    st.sidebar.title("Visualization")
    with st.sidebar:
        page_selectioned = st.radio(
            "Choose the visualization",
            PAGES.keys(),
            index=0
        )
    ClusterData(load_experiment_results(PurePath(experiments_path[0]).parent), load_experiment_config(PurePath(experiments_path[0]).parent)).show_parameters()
    st.divider()
    PAGES[page_selectioned](experiments_path)
else:
    st.header(":red[You don't have experiments to see. (check docs/how_to_visualize_results.md)]")
