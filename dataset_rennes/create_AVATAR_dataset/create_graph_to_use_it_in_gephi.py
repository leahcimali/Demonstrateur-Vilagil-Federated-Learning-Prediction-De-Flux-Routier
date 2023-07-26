"""
Use to create a graph with a JSON file and
save it to gexf format to use it in Gephi software
in order to apply a layout algorithm.
"""


import networkx as nx
import json

# Load the content of the JSON file representing the graph
with open('./data/Rennes_sensors_graph.json') as json_file:
    graph_data = json.load(json_file)

# Create a directed graph from the JSON data
G = nx.DiGraph()

# Add edges to the graph
for edge in graph_data["edges"]:
    G.add_edge(edge["source"], edge["target"])

nx.write_gexf(G, "./graph_avant_gephi.gexf")
