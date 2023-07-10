###############################################################################
# Libraries
###############################################################################
import colorsys
import glob
import math
from pathlib import PurePath
import streamlit as st
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import plotly.graph_objects as go
from utils_data import load_PeMS04_flow_data
from utils_graph import create_graph
from utils_streamlit_app import load_experiment_results, load_experiment_config


random.seed(42)

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
        self.size = len(self.cluster)

    def get_nb_sensor_better_in_federation(self, metric):
        return (
            sum(
                self.cluster[sensor]["Federated"][metric] <= self.cluster[sensor]["local_only"][metric]
                for sensor in self.cluster.keys()
            )
            if metric != "Superior Pred %"
            else sum(
                self.cluster[sensor]["Federated"][metric] >= self.cluster[sensor]["local_only"][metric]
                for sensor in self.cluster.keys()
            )
        )

    def show_parameters(self):
        df_parameters = pd.DataFrame(self.parameters,
                                    columns=["time_serie_percentage_length",
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
        df_parameters = df_parameters.rename(column_names)

        st.write("**Parameters of clusters**")
        st.write("")
        st.write("WS (**Windows size**), how many steps use to make a prediction")
        st.write("PH (**Prediction horizon**), how far the prediction goes (how many steps)")
        st.write("CR (**Communication round**), how many time the central server and the clients communicate")
        df_parameters = pd.DataFrame(df_parameters.iloc[0])
        df_parameters.index.name = "Parameters"
        st.dataframe(df_parameters, use_container_width=True)


def generate_colors(num_colors):
    colors = []
    phi = (1 + math.sqrt(5)) / 2  # Nombre d'or
    for i in range(num_colors):
        couleur_index = int(math.floor(i * phi))
        couleur_r = (couleur_index * 137) % 256
        couleur_g = (couleur_index * 103) % 256
        couleur_b = (couleur_index * 143) % 256
        colors.append(f"rgb({couleur_r}, {couleur_g}, {couleur_b})")
    return colors


def get_couleurs(list_numbers):
    couleurs = {}
    phi = (1 + math.sqrt(5)) / 2  # Nombre d'or

    for nombre in list_numbers:
        if nombre not in couleurs:
            couleur_index = int(math.floor(nombre * phi))
            couleur_r = (couleur_index * 137) % 256
            couleur_g = (couleur_index * 73) % 256
            couleur_b = (couleur_index * 43) % 256
            couleurs[nombre] = f"rgb({couleur_r}, {couleur_g}, {couleur_b})"

    return couleurs


def render_graph_neighborhood(graph):
    st.subheader("Network graph with sensors colored based on their neighborhood")
    pos = nx.spring_layout(graph, k=1, iterations=300, seed=42)
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend((x0, x1, None))
        edge_y.extend((y0, y1, None))
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in graph.nodes():
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
    for adjacencies in graph.adjacency():
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(int(adjacencies[0]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    height=800,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    st.plotly_chart(fig, use_container_width=True)


def render_graph_colored_with_cluster(graph, clusters):
    st.subheader("Network graph with sensors colored based on their cluster membership")
    colors_cluster = generate_colors(28)
    for i in range(len(clusters)):
        color_cluster = colors_cluster[i]
        for sensor in clusters[i].sensors:
            graph.nodes[sensor]["color"] = color_cluster
    for sensor in [222, 79, 90, 2, 81, 204]:
        graph.nodes[sensor]["color"] = colors_cluster[27]

    pos = nx.spring_layout(graph, k=1.0, iterations=300, seed=42)

    fig = go.Figure()
    for edge in graph.edges():
        if graph.nodes[int(edge[0])]["color"] == graph.nodes[int(edge[1])]["color"]:
            group = graph.nodes[int(edge[0])]["color"]
        else:
            group = "other"

        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            legendgroup=group,
            line=dict(width=1.0, color='#888'),
            mode='lines', showlegend=False)
        )

    node_x = []
    node_y = []
    color = []
    for node in graph.nodes():
        x, y = pos[node]
        color.append(graph.nodes[int(node)]["color"])
        node_x.append(x)
        node_y.append(y)
    node_text = [int(adjacencies[0]) for adjacencies in graph.adjacency()]
    for i in range(len(graph.nodes())):
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
        title='',
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        height=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    with st.spinner('Plotting...'):
        st.plotly_chart(fig, use_container_width=True)


def render_tendency_cluster_size_number_improve(clusters, metric, title="", descending=False):
    st.subheader("The relationship between cluster size and the number of sensors improved by the federated approach")
    clusters_group_by_size = {}
    size_clusters_group_by_size = {}
    for cluster in clusters:
        if cluster.size in clusters_group_by_size.keys():
            clusters_group_by_size[cluster.size] += cluster.get_nb_sensor_better_in_federation(metric)
            size_clusters_group_by_size[cluster.size] += 1
        else:
            clusters_group_by_size[cluster.size] = cluster.get_nb_sensor_better_in_federation(metric)
            size_clusters_group_by_size[cluster.size] = 1
    clusters_mean_metric_values = []
    for cluster_size in clusters_group_by_size.keys():
        clusters_group_by_size[cluster_size] = (clusters_group_by_size[cluster_size] / size_clusters_group_by_size[cluster_size])
        clusters_mean_metric_values.append(clusters_group_by_size[cluster_size])

    sizes = clusters_group_by_size.keys()
    sorted_sizes = sorted(sizes, reverse=descending)
    sorted_metric_mean = [value for _, value in sorted(zip(sizes, clusters_mean_metric_values), reverse=descending)]

    max_value = max(clusters_mean_metric_values)

    couleurs = get_couleurs(sorted_sizes)

    fig = go.Figure()
    for (x, y) in zip(sorted_sizes, sorted_metric_mean):
        y = round(y, 2)
        bar_trace = go.Bar(
            x=[x],
            y=[y],
            marker_color=f'{couleurs[x]}',
            name=f'Size of cluster: {x} | Nb sensor: {y}',
            text=f"{y}",
            orientation="v",
        )
        fig.add_trace(bar_trace)

    line_plot = go.Scatter(
        x=sorted_sizes,
        y=sorted_metric_mean,
        marker=dict(
            color="red"
        ),
        name='Tendency',
        mode='lines + markers'
    )
    fig.add_trace(line_plot)

    fig.update_yaxes(
        title='Nb sensors with better results',
        range=[-1, max_value + 2],
        dtick=5
    )

    fig.update_xaxes(
        title='Size of the cluster',
        range=[min(sorted_sizes) - 1, max(sorted_sizes) + 1],
        dtick=1
    )

    fig.update_layout(
        title=f'{title}',
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=0,
        legend=dict(
            title="Size of the cluster and number of sensors better in federated version"
        )
    )

    with st.spinner('Plotting...'):
        st.plotly_chart(fig, use_container_width=True)


def render_proportion_sensor_better_in_federated(clusters, metric, title="", descending=False):
    st.subheader("Compare clusters based on their size and the  proportion of sensors that have better results with the federated version compared to the local version.")
    st.write("""A circle represents a cluster with the size in x""")
    st.write("""The line represents the tendency""")
    clusters_size = [cluster.size for cluster in clusters]

    couleurs = get_couleurs(clusters_size)
    clusters_group_by_size = {}
    size_clusters_group_by_size = {}
    for cluster in clusters:
        if cluster.size in clusters_group_by_size.keys():
            clusters_group_by_size[cluster.size] += cluster.get_nb_sensor_better_in_federation(metric) / cluster.size
            size_clusters_group_by_size[cluster.size] += 1
        else:
            clusters_group_by_size[cluster.size] = cluster.get_nb_sensor_better_in_federation(metric) / cluster.size
            size_clusters_group_by_size[cluster.size] = 1
    clusters_mean_metric = []
    for cluster_size in clusters_group_by_size.keys():
        clusters_group_by_size[cluster_size] = (clusters_group_by_size[cluster_size] / size_clusters_group_by_size[cluster_size]) * 100
        clusters_mean_metric.append(clusters_group_by_size[cluster_size])
    size_group_by_cluster_size = clusters_group_by_size.keys()

    percent_sensor_improve = [
        (cluster.get_nb_sensor_better_in_federation(metric) / cluster.size) * 100
        for cluster in clusters
    ]

    sorted_cluster_size = sorted(clusters_size, reverse=descending)
    sorted_percent_sensor_improve = [value for _, value in sorted(zip(clusters_size, percent_sensor_improve))]

    sorted_size_group_by_cluster_size = sorted(size_group_by_cluster_size, reverse=descending)
    sorted_clusters_mean_metric = [value for _, value in sorted(zip(size_group_by_cluster_size, clusters_mean_metric))]

    max_y_value = max(sorted_percent_sensor_improve)

    fig = go.Figure()
    for (x, y) in zip(sorted_cluster_size, sorted_percent_sensor_improve):
        y = round(y, 2)
        bar_trace = go.Scatter(
            x=[x],
            y=[y],
            marker=dict(
                color=couleurs[x],
                size=20
            ),
            text=f'{y}%',
            name=f'Size: {x} | Percent: {y}',
            legendgroup=x,
        )
        fig.add_trace(bar_trace)

    line_plot = go.Scatter(
        x=sorted_size_group_by_cluster_size,
        y=sorted_clusters_mean_metric,
        marker=dict(
            color="red"
        ),
        name='Tendency',
        mode='lines+markers'
    )
    fig.add_trace(line_plot)

    fig.update_xaxes(title='Size of a cluster',
                    dtick=1)

    fig.update_yaxes(
        title='% of sensors with better results',
        range=[-2, max_y_value + 2],
        dtick=5
    )

    fig.update_layout(
        title=f'{title}',
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_tickangle=0,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_histogram(clusters, metric, title="", descending=False):
    st.subheader("Compare clusters based on their size and the proportion of sensors that have better results with the federated version compared to the local version.")
    st.write("""When 2 or more cluster have the same size we take the average""")
    clusters_group_by_size = {}
    size_clusters_group_by_size = {}
    for cluster in clusters:
        if cluster.size in clusters_group_by_size.keys():
            clusters_group_by_size[cluster.size] += cluster.get_nb_sensor_better_in_federation(metric) / cluster.size
            size_clusters_group_by_size[cluster.size] += 1
        else:
            clusters_group_by_size[cluster.size] = cluster.get_nb_sensor_better_in_federation(metric) / cluster.size
            size_clusters_group_by_size[cluster.size] = 1
    clusters_mean_metric = []
    for cluster_size in clusters_group_by_size.keys():
        clusters_group_by_size[cluster_size] = (clusters_group_by_size[cluster_size] / size_clusters_group_by_size[cluster_size]) * 100
        clusters_mean_metric.append(clusters_group_by_size[cluster_size])
    sizes = clusters_group_by_size.keys()

    sorted_sizes = sorted(sizes, reverse=descending)
    sorted_metric_mean = [size for _, size in sorted(zip(sizes, clusters_mean_metric))]
    couleurs = get_couleurs(sorted_sizes)

    fig = go.Figure()
    for (x, y) in zip(sorted_sizes, sorted_metric_mean):
        y = round(y, 2)
        bar_trace = go.Bar(
            x=[x],
            y=[y],
            marker_color=f'{couleurs[x]}',
            name=f'Size of cluster: {x} Percent: {y}%',
            text=f"{y}",
            orientation="v",
        )
        fig.add_trace(bar_trace)

    fig.add_trace(go.Scatter(
        x=sorted_sizes,
        y=sorted_metric_mean,
        marker=dict(
            color="red"
        ),
        name='Tendency',
        mode='lines+markers'
    ))

    fig.update_xaxes(
        title='Size of the cluster (i)',
        range=[min(sorted_sizes) - 1, max(sorted_sizes) + 1],
        dtick=1
    )

    fig.update_yaxes(
        title='Percent of sensors better in federated',
        range=[0, 100],
        dtick=5
    )

    fig.update_layout(
        title=f'{title}',
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=0,
        legend=dict(
            title="Size of the cluster and percent of sensors better in federated version"
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

_, distance = load_PeMS04_flow_data()
G = create_graph(distance)

with st.spinner('Plotting...'):
    render_graph_neighborhood(G)
    render_graph_colored_with_cluster(G, clusters)
    render_histogram(clusters, "RMSE")
    render_tendency_cluster_size_number_improve(clusters, "RMSE")
    render_proportion_sensor_better_in_federated(clusters, "RMSE")
