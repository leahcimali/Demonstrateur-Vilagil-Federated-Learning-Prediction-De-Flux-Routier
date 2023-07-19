###############################################################################
# Libraries
###############################################################################
from pathlib import PurePath
import streamlit as st
import plotly.graph_objects as go
import networkx as nx

from ClusterData import ClusterData
from utils_streamlit_app import load_experiment_results, load_experiment_config, create_selectbox_metrics, load_graph


#######################################################################
# Function(s)
#######################################################################
def render_bar_plot_fed_vs_local(cluster, metric, sorted_by: str, descending: str):
    st.subheader(f"Comparison between the federated version and local version with the {metric} metric on each sensors")

    value_metric_federated = []
    value_metric_local = []
    for sensor in cluster.indexes:
        value_metric_local.append(cluster.get_sensor_metric_unormalized_local_values(sensor, metric))
        value_metric_federated.append(cluster.get_sensor_metric_unormalized_federated_values(sensor, metric))

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
        hovermode="x unified",
        legend=dict(
            title="Federated version vs Local version"
        )
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


def render_graph(graph, cluster):
    sensors = cluster.sensors_name
    st.subheader("Network graph")
    pos = nx.spring_layout(graph, k=0.1, iterations=200, seed=42)
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        node_0 = edge[0]
        node_1 = edge[1]
        if node_0 in sensors and node_1 in sensors:
            x0, y0 = pos[node_0]
            x1, y1 = pos[node_1]
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
        if node in sensors:
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


#######################################################################
# Main
#######################################################################
st.subheader("Visiualization of one cluster")
st.write("""
        * Visualization of one cluster
        """)
st.divider()


def one_cluster(experiments_path):
    clusters = {}
    for path_exp in experiments_path:
        path_exp_parent = PurePath(path_exp).parent
        cluster = ClusterData(load_experiment_results(path_exp_parent), load_experiment_config(path_exp_parent))
        clusters[cluster.name.split("/")[1]] = cluster

    cluster = st.selectbox("Select the cluster", list(clusters))
    st.subheader(f"Nb sensor in the cluster : {clusters[cluster].size}")
    with st.spinner('Plotting...'):
        G = load_graph()
        render_graph(G, clusters[cluster])

        metric = create_selectbox_metrics()
        col1, col2 = st.columns(2)
        with col1:
            descending = st.radio("Sorted:", ["Descending", "Ascending"], index=1)
        with col2:
            sorted_by = st.radio("Sorted_by:", ["Federated", "Local"], index=0)
        render_bar_plot_fed_vs_local(clusters[cluster], metric, sorted_by, descending=descending)

    def format_selectbox_sensor(value):
        return clusters[cluster].sensors_name[int(value)]

    sensor_selected = st.selectbox('Choose the sensor', clusters[cluster].indexes, format_func=format_selectbox_sensor)
    normalized = st.radio("Normalized data ?", ["Yes", "No"], index=1)
    clusters[cluster].show_results_sensor(sensor_selected, True if normalized == "Yes" else False)
