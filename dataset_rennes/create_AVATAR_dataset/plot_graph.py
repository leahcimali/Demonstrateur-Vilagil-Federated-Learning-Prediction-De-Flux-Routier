"""
Use to plot the graph with plotly after the
graph has been put in good layout with Gephi
software.
"""


import networkx as nx
import plotly.graph_objects as go
import json

# Load the content of the JSON file representing the graph
with open('./data/Rennes_sensors_graph.json') as json_file:
    graph_data = json.load(json_file)

# Create a directed graph from the JSON data
G = nx.DiGraph()

for node in graph_data["nodes"]:
    G.add_node(
        node["key"],
        x=node["attributes"]["x"],
        y=node["attributes"]["y"],
        size=node["attributes"]["size"]
    )
# Add edges to the graph
for edge in graph_data["edges"]:
    G.add_edge(edge["source"], edge["target"], weight=1.0)

# Create a Plotly graph object
edge_x = []
edge_y = []
for edge in G.edges():
    x0 = G.nodes[edge[0]]["x"]
    y0 = G.nodes[edge[0]]["y"]
    x1 = G.nodes[edge[1]]["x"]
    y1 = G.nodes[edge[1]]["y"]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines',
)

node_x = []
node_y = []
for node in G.nodes():
    x = G.nodes[node]["x"]
    y = G.nodes[node]["y"]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    text=list(G.nodes),
    mode='markers+text',
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        showscale=False,
        colorscale='YlGnBu',
        size=10,
        line=dict(width=2)
    )
)


fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
)

fig.show()
