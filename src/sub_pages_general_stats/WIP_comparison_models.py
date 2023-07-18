###############################################################################
# Libraries
###############################################################################
import copy
import glob
import json
from pathlib import PurePath
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode


from utils_streamlit_app import format_radio, get_color_fed_vs_local, load_experiment_config, load_experiment_results, style_dataframe


#######################################################################
# Constant(s)
#######################################################################
METRICS = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]


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


def get_index_with_id(df, id):
    index = df["id"] == id
    return df[index].index[0]


def get_exp_with_id(dict_exp, id):
    return dict_exp[id]


def filter_df(df, source_experiment, filter):
    """filter

    Args:
        df (_type_): _description_
        source_experiment (_type_): _description_
        filter (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_filtered = copy.copy(df)

    # first filter: parameters have to be differents
    for param in filter:
        if param != "Sensor use":
            df_filtered = df_filtered[(df_filtered[param] != source_experiment[param].iloc[0])]

    # second filter: parameters to be the same
    for param in source_experiment.columns:
        if param not in filter and param != "Sensor use" and param != "id":
            df_filtered = df_filtered[(df_filtered[param] == source_experiment[param].iloc[0])]

    # third filter (Very specific)
    for i in df_filtered.index:
        if "Sensor use" in filter:
            if all(x in source_experiment["Sensor use"].iloc[0] for x in df_filtered.loc[i]["Sensor use"]):
                df_filtered = df_filtered.drop([i])
        elif any(x not in source_experiment["Sensor use"].iloc[0] for x in df_filtered.loc[i]["Sensor use"]):
            df_filtered = df_filtered.drop([i])

    # fourth filter (also a specific one)
    df_filtered = df_filtered[(df_filtered["id"] != source_experiment["id"].iloc[0])]
    return df_filtered


#######################################################################
# Main
#######################################################################
def WIP_comparison_models():
    st.subheader("Comparison Models")
    st.write("""
            * On this page select two experiments to compare them.
                * In the table, you will find the general statistics for both the Local version and\\
                the Federated version on differents metrics. On the left the left model and on the\\
                right the other model.
                * In the box plot, you will see the distribution of the RMSE values.
            """)
    st.divider()

    experiments_path = "experiments"
    path_files = glob.glob(f"./{experiments_path}/*/", recursive=True)
    experiments = {}
    experiments_config = []
    for path in path_files:
        path_exp = PurePath(path)
        config = load_experiment_config(path_exp)
        resultats = load_experiment_results(path_exp)
        experiments[config["save_model_path"]] = {
            "config": config,
            "resultats": resultats
        }
        experiments_config.append(config)

    df = pd.DataFrame(experiments_config, columns=["time_serie_percentage_length",
                                                "batch_size",
                                                "number_of_nodes",
                                                "nodes_to_filter",
                                                "window_size",
                                                "prediction_horizon",
                                                "communication_rounds",
                                                "num_epochs_local_federation",
                                                "epoch_local_retrain_after_federation",
                                                "num_epochs_local_no_federation",
                                                "model",
                                                "save_model_path"])
    column_names = {
        "time_serie_percentage_length": "Length of the time serie used",
        "batch_size": "Batch Size",
        "number_of_nodes": "Number of Nodes",
        "nodes_to_filter": "Sensor use",
        "window_size": "WS",
        "prediction_horizon": "PH",
        "communication_rounds": "CR",
        "num_epochs_local_no_federation": "Epochs alone",
        "num_epochs_local_federation": "Epochs Federation",
        "epoch_local_retrain_after_federation": "Epochs Local Retrain",
        "learning_rate": "Learning Rate",
        "model": "Model",
        "save_model_path": "id"
    }

    df = df.rename(columns=column_names)

    st.write("**Configuration explanations:**")
    st.write("*Length of Time Series: Percentage of the time series used before splitting the dataset into train/validation/test sets.")
    st.write("*Window Size (**WS**): The number of time steps in the historical data considered by the model for making predictions.")
    st.write("*Prediction Horizon (**PH**): The number of time steps or observations to forecast beyond the last observation in the input window.")
    st.write("*Communication Round (**CR**): The iteration or cycle of communication between the central server and the actors during the training process.")
    st.write("*Epochs Federation: The number of epochs an actor performs before sending its model to the central server.")
    st.write("*Epochs local alone: The number of epochs used to train the local version, which will be compared to the federated version.")

    gb = GridOptionsBuilder.from_dataframe(df[df.columns])
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gridOptions = gb.build()
    data = AgGrid(
        df,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
    )

    selected_rows = data["selected_rows"]
    if len(selected_rows) > 0:
        index_exp = get_index_with_id(df, selected_rows[0]['id'])
        source_experiment = pd.DataFrame(df.iloc[index_exp]).rename(columns={index_exp: "Value"}).T
        st.dataframe(source_experiment.T, use_container_width=True)

        changing_parameters = list(source_experiment.columns)
        changing_parameters.remove("id")
        selected_changing_parameters_parameters = st.multiselect("Pick parameters that you want to filter other experiments to make comparison", changing_parameters)

        df_filtered = filter_df(df, source_experiment, selected_changing_parameters_parameters)
        st.dataframe(df_filtered, use_container_width=True)

    results_1 = get_exp_with_id(experiments, source_experiment["id"].iloc[0])["resultats"]
    config_1 = get_exp_with_id(experiments, source_experiment["id"].iloc[0])["config"]
    for i in range(len(df_filtered)):
        results_2 = get_exp_with_id(experiments, df_filtered["id"].iloc[i])["resultats"]

        config_2 = get_exp_with_id(experiments, df_filtered["id"].iloc[i])["config"]

        federated_results_model_1 = []
        local_results_model_1 = []
        for i in range(config_1["number_of_nodes"]):
            if "Federated" in results_1["0"].keys():  # e.g. keys = ['Federated', 'local_only']
                federated_results_model_1.append(results_1[str(i)]["Federated"])
            if "local_only" in results_1["0"].keys():  # e.g. keys = ['Federated', 'local_only']
                local_results_model_1.append(results_1[str(i)]["local_only"])

        federated_results_model_1 = pd.DataFrame(federated_results_model_1, columns=METRICS)
        local_results_model_1 = pd.DataFrame(local_results_model_1, columns=METRICS)

        federated_results_model_2 = []
        local_results_model_2 = []
        for j in range(config_2["number_of_nodes"]):
            if "Federated" in results_2["0"].keys():  # e.g. keys = ['Federated', 'local_only']
                federated_results_model_2.append(results_2[str(j)]["Federated"])
            if "local_only" in results_2["0"].keys():  # e.g. keys = ['Federated', 'local_only']
                local_results_model_2.append(results_2[str(j)]["local_only"])

        federated_results_model_2 = pd.DataFrame(federated_results_model_2, columns=METRICS)
        local_results_model_2 = pd.DataFrame(local_results_model_2, columns=METRICS)

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
        for i in range(len(METRICS)):
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
