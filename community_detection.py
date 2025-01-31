from sknetwork.clustering import Louvain
import networkx as nx
import pandas as pd

def labels_from_louvain(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc)

    adjacency = nx.adjacency_matrix(subgraph)
    adjacency_array = adjacency.toarray()
    louvain = Louvain()
    _ = louvain.fit_transform(adjacency_array)

    return louvain.labels_.tolist()

def communities_from_louvain(graph):
    labels = labels_from_louvain(graph)
    communities = {}
    for node, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    return communities

def unique_attributes_in_communities(graph, attributes, communities):
    """
    Detects unique values of attributes in communities
    """
    unique_attributes = {}
    for community, nodes in communities.items():
        unique_attributes[community] = {}
        for attribute in attributes:
            values = set()
            for node in nodes:
                values.add(graph.nodes[str(node)][attribute])
            unique_attributes[community][attribute] = values
    return unique_attributes

def prop_attributes_in_communities(graph, communities, unique_attributes):
    """
    Returns the proportion of each value of each attribute in each community
    """
    prop_attributes = {}
    for community, attributes_values in unique_attributes.items():
        prop_attributes[community] = {"community_size": len(communities[community])}
        for attribute, values in attributes_values.items():
            prop_attributes[community][attribute] = {}
            for value in values:
                prop_attributes[community][attribute][value] = 0
                for node in communities[community]:
                    if graph.nodes[str(node)][attribute] == value:
                        prop_attributes[community][attribute][value] += 1
                prop_attributes[community][attribute][value] /= len(communities[community])
                # Make the keys ordered in descending order of proportion
                prop_attributes[community][attribute] = dict(sorted(prop_attributes[community][attribute].items(), key=lambda item: item[1], reverse=True))
                # Round the proportions to 4 decimal places
                prop_attributes[community][attribute] = {k: round(v, 4) for k, v in prop_attributes[community][attribute].items()}
                # Keep only top 5 values
                prop_attributes[community][attribute] = dict(list(prop_attributes[community][attribute].items())[:5])
                
    return prop_attributes

def correlation_attributes_in_communities(graph, communities, unique_attributes):
    """
    Returns the correlation between each pair of attributes in each community
    """
    correlation_attributes = {}
    for community, attributes_values in unique_attributes.items():
        correlation_attributes[community] = {}
        for attribute1 in attributes_values.keys():
            # correlation_attributes[community][attribute1] = {}
            for attribute2 in attributes_values.keys():
                if attribute1 != attribute2:
                    correlation_attributes[community][(attribute1,attribute2)] = 0
                    for node in communities[community]:
                        if graph.nodes[str(node)][attribute1] == graph.nodes[str(node)][attribute2]:
                            correlation_attributes[community][(attribute1,attribute2)] += 1
                    correlation_attributes[community][(attribute1,attribute2)] /= len(communities[community])
                    # Round the correlation to 4 decimal places
                    correlation_attributes[community][(attribute1,attribute2)] = round(correlation_attributes[community][(attribute1,attribute2)], 4)
                    # Remove duplicate pairs
                    if (attribute2, attribute1) in correlation_attributes[community]:
                        del correlation_attributes[community][(attribute2, attribute1)]

    return correlation_attributes