###############################################################################
# Libraries
###############################################################################
import glob
import math
from pathlib import PurePath
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from utils_data import load_PeMS04_flow_data
from utils_graph import create_graph


from utils_streamlit_app import load_experiment_results, load_experiment_config


#######################################################################
# Constant(s)
#######################################################################
METRICS = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]


#######################################################################
# Function(s)
#######################################################################
class Cluster:
    def __init__(self, cluster, config_cluster):
        super(Cluster, self).__init__()
        self.cluster = cluster
        self.parameters = config_cluster
        self.nodes = cluster.keys()
        self.sensors = config_cluster["nodes_to_filter"]
        self.name = config_cluster["save_model_path"]
        self.size = len(cluster)

    def get_node_metric_local(self, node, metric):
        return self.cluster[node]["local_only"][metric]

    def get_node_metric_federated(self, node, metric):
        return self.cluster[node]["Federated"][metric]


    def show_parameters(self):
        # Création du DataFrame
        df_parameters = pd.DataFrame(self.parameters, columns=["time_serie_percentage_length",
                                                    "batch_size",
                                                    "number_of_nodes",
                                                    "nodes_to_filter",
                                                    "window_size",
                                                    "prediction_horizon",
                                                    "communication_rounds",
                                                    "num_epochs_local_federation",
                                                    "epoch_local_retrain_after_federation",
                                                    "num_epochs_local_no_federation",
                                                    "model"])
        # Renommage des colonnes
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
            "epoch_local_retrain_after_federation": "Epoch Local Retrain",
            "learning_rate": "Learning Rate",
            "model": "Model"
        }

        # Affichage du tableau récapitulatif
        st.write("**Parameters of clusters**")
        st.write("")
        st.write("WS (**Windows size**), how many steps use to make a prediction")
        st.write("PH (**Prediction horizon**), how far the prediction goes (how many steps)")
        st.write("CR (**Communication round**), how many time the central server and the clients communicate")
        df_parameters = pd.DataFrame(df_parameters.iloc[0])
        df_parameters.index.name = "Parameters"
        df_parameters = df_parameters.rename(column_names)
        st.dataframe(df_parameters, use_container_width=True)


def render_bar_plot_fed_vs_local(cluster, metric, descending=False):


    value_metric_local = []
    value_metric_federated = []
    for node in cluster.nodes:
        value_metric_local.append(cluster.get_node_metric_local(node, metric))
        value_metric_federated.append(cluster.get_node_metric_federated(node, metric))

    sorted_value_metric_federated = sorted(value_metric_federated, reverse=descending)
    sorted_value_metric_local = [value for _, value in sorted(zip(value_metric_federated, value_metric_local))]
    sorted_sensor_name = [str(name) for _, name in sorted(zip(value_metric_federated, cluster.sensors))]

    max_value = max(sorted_value_metric_federated, sorted_value_metric_local)

    fig = go.Figure()
    for (sensor, federated_value, local_value) in zip(sorted_sensor_name, sorted_value_metric_federated, sorted_value_metric_local):
        federated_value = round(federated_value, 2)
        local_value = round(local_value, 2)
        bar_trace_federated = go.Bar(
            x=[f"sensor: {sensor}"],
            y=[federated_value],
            name=f'{sensor} Federated {metric} = {federated_value}',
            orientation="v",
            marker_color="rgb(120, 120, 120)",
            width=0.2,
            offset=-0.12
        )
        bar_trace_local = go.Bar(
            x=[f"sensor: {sensor}"],
            y=[local_value],
            name=f'{sensor} Local {metric} = {local_value}',
            orientation="v",
            marker_color="rgb(220, 120, 220)",
            width=0.2,
            offset=0.12
        )
        fig.add_trace(bar_trace_federated)
        fig.add_trace(bar_trace_local)

    # Ajouter un thème sombre au graphique
    fig.update_layout(template="plotly_dark")

    # Ajouter un titre au graphique
    fig.update_layout(title=f"Comparison between federated version and local version on the {metric} metric")

    # Ajouter des légendes aux axes x et y
    fig.update_xaxes(title="Nom du capteur")
    fig.update_yaxes(title=f"{metric} value", range=[0, max_value], dtick=5)

    fig.update_layout(
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=45,
    )

    # Utiliser le mode de survol "x unified" pour le graphique
    fig.update_layout(hovermode="x unified")
    nb_sensors = sum(
        federated_value <= local_value
        for federated_value, local_value in zip(
            value_metric_federated, value_metric_local
        )
    )

    # Ajouter une annotation au graphique avec le nombre de capteurs calculé
    fig.add_annotation(
        x=0,
        y=1,
        text=f" {nb_sensors} sensor(s) is/are better with the federation on {len(value_metric_federated)} ({round(nb_sensors / len(value_metric_federated)*100, 2)}%)",
        font=dict(size=18, color="black"),
        showarrow=False,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        bordercolor="red",
        borderwidth=5,
        bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)


#######################################################################
# Main
#######################################################################
st.subheader("Comparison Clusters")
st.write("""
        * TODO
        """)
st.divider()

# Chargement des configurations des expérimentations
experiments_folder = "community_experiment/"  # Chemin où toutes les expérimentations sont enregistrées
experiments_path = glob.glob(f"./{experiments_folder}**/config.json", recursive=True)

clusters = {}
for path_exp in experiments_path:
    path_exp_parent = PurePath(path_exp).parent
    cluster = Cluster(load_experiment_results(path_exp_parent), load_experiment_config(path_exp_parent))
    clusters[cluster.name.split("/")[1]] = cluster

cluster = st.selectbox("Select the cluster", [name for name in clusters])
clusters[cluster].show_parameters()
render_bar_plot_fed_vs_local(clusters[cluster], "RMSE")
