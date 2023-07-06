###############################################################################
# Libraries
###############################################################################
import glob
import math
from pathlib import PurePath
import streamlit as st
import pandas as pd
import networkx as nx
import random


import plotly.graph_objects as go
from utils_data import load_PeMS04_flow_data
from utils_graph import create_graph
from utils_streamlit_app import load_experiment_results, load_experiment_config


random.seed(5)

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
        self.sensors = [config_cluster["nodes_to_filter"][int(index)] for index in cluster.keys()]
        self.name = str(config_cluster["nodes_to_filter"])
        self.size = len(cluster)
        self.general_stats_local_only = pd.DataFrame([
            cluster[sensor]["local_only"]
            for sensor in cluster.keys()
        ]).describe().T
        self.general_stats_federated = pd.DataFrame([
            cluster[sensor]["Federated"]
            for sensor in cluster.keys()
        ]).describe().T

    def get_general_metric_mean_local(self, metric):
        """Return the general average for the local version of a choosen metric

        Args:
            metric (_type_): _description_

        Returns:
            _type_: _description_
        """
        st.write(self.general_stats_local_only)
        return self.general_stats_local_only.loc[metric]["mean"].item()

    def get_general_metric_mean_federated(self, metric):
        """Return the general average for the federated version of a choosen metric

        Args:
            metric (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.general_stats_federated.loc[metric]["mean"].item()

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


def render_graph(graph):
    pos = nx.spring_layout(G, k=0.4, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node)

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Network graph with sensors colored based on their neighbors',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    height=800,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    # Display the graph with Streamlit
    st.plotly_chart(fig, use_container_width=True)


def create_color_list(n):
    # n is the number of colors needed
    # create an empty list to store the colors
    color_list = []
    # loop n times
    for _ in range(n):
        # generate a random color in rgb format
        color = "rgb(" + ",".join([str(random.randint(0, 255)) for _ in range(3)]) + ")"
        # append the color to the list
        color_list.append(color)
    # return the list of colors
    return color_list


def render_graph_colored_with_cluster(graph, clusters):
    colors_cluster = create_color_list(28)
    for i in range(len(clusters)):
        color_cluster = colors_cluster[i]
        for node in clusters[i].sensors:
            graph.nodes[node]["color"] = color_cluster
    for node in [222.0, 79.0, 90.0, 2.0, 81.0, 204.0]:
        graph.nodes[node]["color"] = colors_cluster[i]

    pos = nx.spring_layout(G, k=0.4, seed=42)

    node_adjacencies = []
    node_text = []
    for adjacencies in graph.adjacency():
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(int(adjacencies[0]))

    node_x = []
    node_y = []
    color = []
    for node in G.nodes():
        x, y = pos[node]
        color.append(graph.nodes[int(node)]["color"])
        node_x.append(x)
        node_y.append(y)

    fig = go.Figure()
    for edge in G.edges():
        if G.nodes[edge[0]]["color"] == G.nodes[edge[1]]["color"]:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                legendgroup=G.nodes[edge[0]]["color"],
                line=dict(width=1.0, color='#888'),
                mode='lines', showlegend=False)
            )
        else:
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                legendgroup="other",
                line=dict(width=1.0, color='#888'),
                mode='lines', showlegend=False)
            )
    for i in range(len(node_x)):
        fig.add_trace(
            go.Scatter(
                x=[node_x[i]],
                y=[node_y[i]],
                mode='markers',
                hoverinfo='text',
                legendgroup=color[i],
                name=node_text[i],
                text=node_text[i],
                marker_color=color[i],
                marker=dict(
                    size=10,
                    line_width=2
                )
            ))

    fig.update_layout(
        title='Network graph with sensors colored based on their cluster membership',
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        height=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_bar_plot(clusters, metric, title="Bar plot", descending=False):
    clusters_name = [cluster.name for cluster in clusters]
    clusters_mean_metric = [cluster.get_general_metric_mean_federated(metric) for cluster in clusters]
    num_clusters = len(clusters_name)
    couleurs = {}
    for i in range(num_clusters):
        proportion = i / num_clusters
        angle = proportion * 2 * math.pi
        r = math.floor(math.sin(angle) * 127) + 128
        g = math.floor(math.sin(angle + 2 * math.pi / 3) * 127) + 128
        b = math.floor(math.sin(angle + 4 * math.pi / 3) * 127) + 128
        couleur = (f"rgb({r}, {g}, {b})")
        cluster_name = clusters_name[i]
        couleurs[cluster_name] = couleur

    sorted_metric_mean = sorted(clusters_mean_metric, reverse=descending)
    sorted_nodes_to_filter = [name for _, name in sorted(zip(clusters_mean_metric, clusters_name))]

    max_value = max(sorted_metric_mean)

    fig = go.Figure()
    for (x, y) in zip(sorted_nodes_to_filter, sorted_metric_mean):
        y = round(y, 2)
        fig.add_trace(go.Bar(
            x=[x],
            y=[y],
            marker_color=f'{couleurs[x]}',
            name=f'{x} = {y}',
            orientation="v",
        ))

    # Configuration de l'axe x
    fig.update_xaxes(title='sensors',
                    dtick=1)

    fig.update_yaxes(
        title=f'{metric} Mean',
        range=[0, max_value],
        dtick=5
    )

    fig.update_layout(
        title=f'{title}',
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=45,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_proportion_sensor_better_in_federated(clusters, metric, title="Bar plot", descending=False):
    clusters_name = [cluster.name for cluster in clusters]
    num_clusters = len(clusters_name)
    couleurs = {}
    for i in range(num_clusters):
        proportion = i / num_clusters
        angle = proportion * 2 * math.pi
        r = math.floor(math.sin(angle) * 127) + 128
        g = math.floor(math.sin(angle + 2 * math.pi / 3) * 127) + 128
        b = math.floor(math.sin(angle + 4 * math.pi / 3) * 127) + 128
        couleur = (f"rgb({r}, {g}, {b})")
        cluster_name = clusters_name[i]
        couleurs[cluster_name] = couleur

    nb_sensor_improve = []
    cluster_size = {}
    for cluster in clusters:
        sensor_improve = 0
        for sensor in cluster.cluster.keys():
            if cluster.cluster[sensor]["Federated"][metric] >= cluster.cluster[sensor]["local_only"][metric]:
                sensor_improve += 1
        nb_sensor_improve.append((sensor_improve / cluster.size) * 100)
        cluster_size[cluster.name] = cluster.size

    sorted_nb_sensor_improve = sorted(nb_sensor_improve, reverse=descending)
    sorted_cluster_name = [name for _, name in sorted(zip(nb_sensor_improve, clusters_name))]

    max_value = max(nb_sensor_improve)

    fig = go.Figure()
    for (x, y) in zip(sorted_cluster_name, sorted_nb_sensor_improve):
        y = round(y, 2)
        bar_trace = go.Bar(
            x=[x],
            y=[y],
            marker_color=f'{couleurs[x]}',
            name=f'{x} = {y}',
            text=f"{cluster_size[x]}",
            orientation="v",
        )
        fig.add_trace(bar_trace)

    # Configuration de l'axe x
    fig.update_xaxes(title='sensors',
                    dtick=1)

    fig.update_yaxes(
        title=f'{metric} Mean',
        range=[0, max_value],
        dtick=5
    )

    fig.update_layout(
        title=f'{title}',
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=45,
    )

    # Affichage du graphique
    st.plotly_chart(fig, use_container_width=True)


def render_group_by_size(clusters, metric, title="Bar plot", descending=False):
    clusters_group_by_size = {}
    size_clusters_group_by_size = {}
    for cluster in clusters:
        if cluster.size in clusters_group_by_size.keys():
            clusters_group_by_size[cluster.size] += cluster.get_general_metric_mean_federated(metric)
            size_clusters_group_by_size[cluster.size] += 1
        else:
            clusters_group_by_size[cluster.size] = cluster.get_general_metric_mean_federated(metric)
            size_clusters_group_by_size[cluster.size] = 1
    clusters_mean_metric = []
    for cluster_size in clusters_group_by_size.keys():
        clusters_group_by_size[cluster_size] = clusters_group_by_size[cluster_size] / size_clusters_group_by_size[cluster_size]
        clusters_mean_metric.append(clusters_group_by_size[cluster_size])
    sizes = size_clusters_group_by_size.keys()
    sorted_sizes = sorted(sizes, reverse=descending)
    sorted_metric_mean = [metric_value for _, metric_value in sorted(zip(sizes, clusters_mean_metric))]

    max_value = max(sorted_metric_mean)
    fig = go.Figure()
    for (x, y) in zip(sorted_sizes, sorted_metric_mean):
        y = round(y, 2)
        bar_trace = go.Bar(
            x=[x],
            y=[y],
            marker_color=f'rgb({((y * 7))}, 100, 200)',
            name=f'{x, "RMSE:",y}',
            text=f"{y}",
            orientation="v",
        )
        fig.add_trace(bar_trace)

    # Configuration de l'axe x
    fig.update_xaxes(title='Size of the cluster',
                    dtick=1)

    fig.update_yaxes(
        title=f'{metric} Mean',
        range=[0, max_value],
        dtick=5
    )

    fig.update_layout(
        title=f'{title}',
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=0,
        legend=dict(
            title="Size of the cluster"
        )
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

clusters = []
for path_exp in experiments_path:
    path_exp_parent = PurePath(path_exp).parent
    cluster = Cluster(load_experiment_results(path_exp_parent), load_experiment_config(path_exp_parent))
    clusters.append(cluster)

clusters[0].show_parameters()

df_PeMS, distance = load_PeMS04_flow_data()
G = create_graph(distance)
render_graph(G)
render_graph_colored_with_cluster(G, clusters)
render_bar_plot(clusters, "RMSE")
render_proportion_sensor_better_in_federated(clusters, "RMSE")
render_group_by_size(clusters, "RMSE")
