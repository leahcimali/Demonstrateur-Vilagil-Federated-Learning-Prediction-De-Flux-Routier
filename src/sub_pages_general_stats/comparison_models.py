###############################################################################
# Libraries
###############################################################################
import glob
import json
from pathlib import PurePath
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


from utils_streamlit_app import format_radio, get_color_fed_vs_local, style_dataframe


#######################################################################
# Function(s)
#######################################################################
def compare_config(path_file_1, path_file_2):
    with open(f"{path_file_1}/config.json") as f:
        config_1 = json.load(f)
    with open(f"{path_file_2}/config.json") as f:
        config_2 = json.load(f)

    return config_1["nodes_to_filter"] == config_2["nodes_to_filter"]


def selection_of_experiments_custom():
    experiments = "experiments/"  # PATH where all the experiments are saved
    if path_files := glob.glob(f"./{experiments}**/config.json", recursive=True):
        col1_model_1, col2_model_2 = st.columns(2)
        with col1_model_1:
            model_1 = st.radio(
                "Choose the first model",
                path_files, key="model_1", format_func=format_radio)

        with col2_model_2:
            model_2 = st.radio(
                "Choose the second model",
                path_files, key="model_2", format_func=format_radio)
        if (model_1 == model_2):
            st.header(":red[You choose the same experiment]")
            return None
        return [PurePath(model_1).parent, PurePath(model_2).parent]
    return None


def box_plot_comparison(serie_1, serie_2, name_1: str, name_2: str, title: str, xaxis_title: str, yaxis_title: str):
    fig = go.Figure()
    box_1 = go.Box(y=serie_1, marker_color="grey", boxmean='sd', name=name_1, boxpoints=False)
    box_2 = go.Box(y=serie_2, marker_color="orange", boxmean='sd', name=name_2, boxpoints=False)
    fig.add_trace(box_1)
    fig.add_trace(box_2)
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title=f"{xaxis_title}",
        yaxis_title=f"{yaxis_title}",
        #  yaxis=dict(range=[0, max_y_value]),
        font=dict(
            size=28,
            color="#FF7f7f"
        ),
        height=900, width=350
    )
    fig.update_traces(jitter=0)
    return fig


#######################################################################
# Main
#######################################################################
def comparison_models():
    st.subheader("Comparison Models")
    st.write("""
            * On this page select two experiments to compare them.
                * In the table, you will find the general statistics for both the Local version and\\
                the Federated version on differents metrics. On the left the left model and on the\\
                right the other model.
                * In the box plot, you will see the distribution of the RMSE values.
            """)
    st.divider()

    paths_experiment_selected = selection_of_experiments_custom()
    if (paths_experiment_selected is not None):
        path_model_1 = paths_experiment_selected[0]
        path_model_2 = paths_experiment_selected[1]

        with open(f"{path_model_1}/test.json") as f:
            results_1 = json.load(f)
        with open(f"{path_model_1}/config.json") as f:
            config_1 = json.load(f)

        with open(f"{path_model_2}/test.json") as f:
            results_2 = json.load(f)
        with open(f"{path_model_2}/config.json") as f:
            config_2 = json.load(f)

        metrics = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]

        federated_results_model_1 = []
        local_results_model_1 = []
        for i in range(config_1["number_of_nodes"]):
            if "Federated" in results_1["0"].keys():  # e.g. keys = ['Federated', 'local_only']
                federated_results_model_1.append(results_1[str(i)]["Federated"])
            if "local_only" in results_1["0"].keys():  # e.g. keys = ['Federated', 'local_only']
                local_results_model_1.append(results_1[str(i)]["local_only"])

        federated_results_model_1 = pd.DataFrame(federated_results_model_1, columns=metrics)
        local_results_model_1 = pd.DataFrame(local_results_model_1, columns=metrics)

        federated_results_model_2 = []
        local_results_model_2 = []
        for j in range(config_2["number_of_nodes"]):
            if "Federated" in results_2["0"].keys():  # e.g. keys = ['Federated', 'local_only']
                federated_results_model_2.append(results_2[str(j)]["Federated"])
            if "local_only" in results_2["0"].keys():  # e.g. keys = ['Federated', 'local_only']
                local_results_model_2.append(results_2[str(j)]["local_only"])

        federated_results_model_2 = pd.DataFrame(federated_results_model_2, columns=metrics)
        local_results_model_2 = pd.DataFrame(local_results_model_2, columns=metrics)

        _, c2_title_df, _ = st.columns((2, 1, 2))

        with c2_title_df:
            st.header("Sensor in Federation/Local")

        c1_model_1, c2_model_2 = st.columns(2)

        federated_results_model_1_stats = federated_results_model_1.describe().T
        local_results_model_1_stats = local_results_model_1.describe().T
        federated_results_model_2_stats = federated_results_model_2.describe().T
        local_results_model_2_stats = local_results_model_2.describe().T

        color_fed_model_1 = []
        color_local_model_1 = []
        color_fed_model_2 = []
        color_local_model_2 = []
        for i in range(len(metrics)):
            if (i < 3):  # because "Superior Pred %" metric needs to be superior=True
                col_fed_model_1, col_fed_model_2 = get_color_fed_vs_local(federated_results_model_1_stats.iloc[i]["mean"], federated_results_model_2_stats.iloc[i]["mean"], superior=False)
                col_local_model_1, col_local_model_2 = get_color_fed_vs_local(local_results_model_1_stats.iloc[i]["mean"], local_results_model_2_stats.iloc[i]["mean"], superior=False)
            else:
                col_fed_model_1, col_fed_model_2 = get_color_fed_vs_local(federated_results_model_1_stats.iloc[i]["mean"], federated_results_model_2_stats.iloc[i]["mean"], superior=True)
                col_local_model_1, col_local_model_2 = get_color_fed_vs_local(local_results_model_1_stats.iloc[i]["mean"], local_results_model_2_stats.iloc[i]["mean"], superior=True)
            color_fed_model_1.append(col_fed_model_1)
            color_local_model_1.append(col_local_model_1)
            color_fed_model_2.append(col_fed_model_2)
            color_local_model_2.append(col_local_model_2)
        #######################################################################
        # Model 1
        #######################################################################
        with c1_model_1:
            model_1_name = config_1["model"]
            st.divider()
            st.subheader(f"{model_1_name}")
            st.divider()

            st.subheader("Federated Version")
            st.table(federated_results_model_1_stats.style.set_table_styles(style_dataframe(federated_results_model_1_stats, colors=color_fed_model_1, column_index=3)).format("{:.2f}"))
            st.subheader("Local Version")
            st.table(local_results_model_1_stats.style.set_table_styles(style_dataframe(local_results_model_1_stats, colors=color_local_model_1, column_index=3)).format("{:.2f}"))
            st.plotly_chart(
                box_plot_comparison(federated_results_model_1["RMSE"],
                                    local_results_model_1["RMSE"],
                                    "Federated",
                                    "Local",
                                    config_1["model"],
                                    "Version",
                                    "RMSE Values"),
                use_container_width=True)

        #######################################################################
        # Model 2
        #######################################################################
        with c2_model_2:
            model_2_name = config_2["model"]
            st.divider()
            st.subheader(f"{model_2_name}")
            st.divider()
            st.subheader("Federated Version")
            st.table(federated_results_model_2_stats.style.set_table_styles(style_dataframe(federated_results_model_2_stats, colors=color_fed_model_2, column_index=3)).format("{:.2f}"))
            st.subheader("Local Version")
            st.table(local_results_model_2_stats.style.set_table_styles(style_dataframe(local_results_model_2_stats, colors=color_local_model_2, column_index=3)).format("{:.2f}"))
            st.plotly_chart(
                box_plot_comparison(federated_results_model_2["RMSE"],
                                    local_results_model_2["RMSE"],
                                    "Federated",
                                    "Local",
                                    config_2["model"],
                                    "Version",
                                    "RMSE Values"),
                use_container_width=True)
