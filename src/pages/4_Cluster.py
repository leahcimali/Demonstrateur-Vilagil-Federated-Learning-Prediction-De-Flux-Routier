###############################################################################
# Libraries
###############################################################################
import glob
from pathlib import PurePath
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


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
        self.sensors = cluster.keys()
        self.sensors_name = config_cluster["nodes_to_filter"]
        self.name = config_cluster["save_model_path"]
        self.size = len(cluster)

    def get_sensor_metric_local_values(self, node, metric):
        return self.cluster[node]["local_only"][metric]

    def get_sensor_metric_federated_values(self, node, metric):
        return self.cluster[node]["Federated"][metric]

    def get_nb_sensor_better_in_federation(self, metric):
        nb_sensor = 0
        if metric != "Superior Pred %":
            for sensor in self.sensors:
                if self.cluster[sensor]["Federated"][metric] <= self.cluster[sensor]["local_only"][metric]:
                    nb_sensor += 1
        else:
            for sensor in self.sensors:
                if self.cluster[sensor]["Federated"][metric] >= self.cluster[sensor]["local_only"][metric]:
                    nb_sensor += 1
        return nb_sensor

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
                                                    "model"]).iloc[0]
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
        st.subheader("Parameters of the cluster")
        st.write("Note: only the number of sensor and the sensors use in the cluster change between clusters.")
        st.write("")
        st.write("WS (**Windows size**), how many steps use to make a prediction")
        st.write("PH (**Prediction horizon**), how far the prediction goes (how many steps)")
        st.write("CR (**Communication round**), how many time the central server and the clients communicate")
        df_parameters.index.name = "Parameters"
        df_parameters = df_parameters.rename(column_names)
        st.dataframe(df_parameters, use_container_width=True)


def render_bar_plot_fed_vs_local(cluster, metric, sorted_by: str, descending: str):
    st.subheader(f"Comparison between the federated version and local version with the {metric} metric on each sensors")

    value_metric_federated = []
    value_metric_local = []
    for sensor in cluster.sensors:
        value_metric_local.append(cluster.get_sensor_metric_local_values(sensor, metric))
        value_metric_federated.append(cluster.get_sensor_metric_federated_values(sensor, metric))

    descending = descending == "Descending"
    if sorted_by == "Federated":
        sorted_value_metric_federated = sorted(value_metric_federated, reverse=descending)
        sorted_value_metric_local = [value for _, value in sorted(zip(value_metric_federated, value_metric_local), reverse=descending)]
    else:
        sorted_value_metric_local = sorted(value_metric_local, reverse=descending)
        sorted_value_metric_federated = [value for _, value in sorted(zip(value_metric_local, value_metric_federated), reverse=descending)]

    sorted_sensor_name = [str(name) for _, name in sorted(zip(value_metric_federated, cluster.sensors_name))]

    max_value = max(sorted_value_metric_federated, sorted_value_metric_local)

    fig = go.Figure()
    for (sensor, federated_value, local_value) in zip(sorted_sensor_name, sorted_value_metric_federated, sorted_value_metric_local):
        federated_value = round(federated_value, 2)
        local_value = round(local_value, 2)
        bar_trace_federated = go.Bar(
            x=[f"sensor: {sensor}"],
            y=[federated_value],
            name=f'{sensor} Federated | {metric} = {federated_value}',
            orientation="v",
            marker_color="rgb(50, 50, 64)",
            legendgroup=sensor,
            width=0.2,
            offset=-0.13
        )
        bar_trace_local = go.Bar(
            x=[f"sensor: {sensor}"],
            y=[local_value],
            name=f'{sensor} Local | {metric} = {local_value}',
            orientation="v",
            marker_color="rgb(240, 100, 200)",
            legendgroup=sensor,
            width=0.2,
            offset=0.13
        )
        fig.add_trace(bar_trace_federated)
        fig.add_trace(bar_trace_local)

    fig.update_xaxes(title="Sensors name")
    fig.update_yaxes(title=f"{metric} value", range=[0, max_value], dtick=5)

    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=45,
        hovermode="x unified"
    )

    nb_sensors = cluster.get_nb_sensor_better_in_federation(metric)

    fig.add_annotation(
        x=0,
        y=1,
        text=f"{round(nb_sensors / cluster.size * 100, 2)}% ({nb_sensors}/{cluster.size}) sensors have better results with the federated approach",
        font=dict(size=18, color="black"),
        showarrow=False,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        bordercolor="red",
        borderwidth=3,
        bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)


#######################################################################
# Main
#######################################################################
st.subheader("Visiualization of one cluster")
st.write("""
        * Visualization of one cluster
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

cluster = st.selectbox("Select the cluster", list(clusters))
clusters[cluster].show_parameters()

metric = st.selectbox("Choose the metric", METRICS)
col1, col2 = st.columns(2)
with col1:
    descending = st.radio("Sorted:", ["Descending", "Ascending"], index=1)
with col2:
    sorted_by = st.radio("Sorted_by:", ["Federated", "Local"], index=1)
render_bar_plot_fed_vs_local(clusters[cluster], metric, sorted_by, descending=descending)