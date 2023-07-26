###############################################################################
# Libraries
###############################################################################
import math
from pathlib import PurePath
import streamlit as st
import networkx as nx
import random


import plotly.graph_objects as go
from ClusterData import ClusterData
from utils_streamlit_app import load_experiment_results, load_experiment_config, load_graph


random.seed(42)


#######################################################################
# Function(s)
#######################################################################
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
    for sensor in [222, 79, 90, 2, 81, 204]:  # replace this when the experiment on this cluster finished
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


def render_proportion_sensor_better_in_federated_no_mean(clusters, metric, title="", descending=False):
    st.subheader("Compare clusters based on their size and the  proportion of sensors that have better results with the federated version compared to the local version")
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


def render_proportion_sensor_better_in_federated(clusters, metric, title="", descending=False):
    st.subheader("Compare clusters based on their size and the proportion of sensors that have better results with the federated version compared to the local version")
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


def render_bar_plot_comparison_federated(clusters, metric, title="Bar plot", descending=False):
    st.subheader("Comparsion between clusters on the :red[Federated version] based on the metric RMSE")
    normalize = st.radio("Normalized data ?", ["Yes", "No"], key="bar_plot_comparison_federated")
    if normalize == "Yes":
        normalized = True
    else:
        normalized = False
    clusters_name = [str(cluster.sensors_name) for cluster in clusters]
    clusters_mean_metric = [cluster.get_sensors_federated_stats(metric, normalized=normalized) for cluster in clusters]
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
    fig.update_xaxes(title='sensors')

    fig.update_yaxes(
        title=f'{metric} Mean',
        range=[0, max_value],
        dtick=5 if normalize == "No" else 0.1
    )

    fig.update_layout(
        title=f'{title}',
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=45,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_bar_plot_comparison_local(clusters, metric, title="Bar plot", descending=False):
    st.subheader("Comparsion between clusters on the :red[Local version] based on the metric RMSE")
    normalize = st.radio("Normalized data ?", ["Yes", "No"], key="bar_plot_comparison_local")
    if normalize == "Yes":
        normalized = True
    else:
        normalized = False
    clusters_name = [str(cluster.sensors_name) for cluster in clusters]
    clusters_mean_metric = [cluster.get_sensors_local_stats(metric, normalized=normalized) for cluster in clusters]
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
    fig.update_xaxes(title='sensors')

    fig.update_yaxes(
        title=f'{metric} Mean',
        range=[0, max_value],
        dtick=5 if normalize == "No" else 0.1
    )

    fig.update_layout(
        title=f'{title}',
        showlegend=True,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_tickangle=45,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_bar_plot_proportion_sensor_better_in_federated_no_mean(clusters, metric, title="", descending=False):
    st.subheader("Compare clusters based on their proportion of sensors that have better results with the federated version compared to the local version")
    clusters_name = [str(cluster.sensors_name) for cluster in clusters]
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

    clusters_proportion_better_in_federation = []
    for cluster in clusters:
        clusters_proportion_better_in_federation.append((cluster.get_nb_sensor_better_in_federation(metric) / cluster.size) * 100)

    sorted_percent_sensor_improve = sorted(clusters_proportion_better_in_federation, reverse=descending)
    sorted_name = [value for _, value in sorted(zip(clusters_proportion_better_in_federation, clusters_name))]

    max_y_value = max(sorted_percent_sensor_improve)

    fig = go.Figure()
    for (x, y) in zip(sorted_name, sorted_percent_sensor_improve):
        y = round(y, 2)
        bar_trace = go.Bar(
            x=[x],
            y=[y],
            marker_color=couleurs[x],
            text=f'{y}%',
            name=f'Name: {x}',
        )
        fig.add_trace(bar_trace)

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
        xaxis_tickangle=45,
    )

    st.plotly_chart(fig, use_container_width=True)


#######################################################################
# Main
#######################################################################
def all_clusters(experiments_path):
    st.subheader("All Clusters")
    st.write("""* TODO""")
    clusters = []
    for path_exp in experiments_path:
        path_exp_parent = PurePath(path_exp).parent
        cluster = ClusterData(load_experiment_results(path_exp_parent), load_experiment_config(path_exp_parent))
        clusters.append(cluster)

    st.subheader(f"Nb cluster(s): {len(clusters)}")
    with st.spinner('Plotting...'):
        G = load_graph()
        render_graph_neighborhood(G)
        render_graph_colored_with_cluster(G, clusters)
        render_bar_plot_comparison_federated(clusters, "RMSE", "", descending=False)
        render_bar_plot_comparison_local(clusters, "RMSE", "", descending=False)
        render_bar_plot_proportion_sensor_better_in_federated_no_mean(clusters, "RMSE", "")
        render_proportion_sensor_better_in_federated(clusters, "RMSE")
        render_proportion_sensor_better_in_federated_no_mean(clusters, "RMSE")
