import numpy as np
import networkx as nx
import pandas as pd

def fisher_yates(arr):
    """
    Parameters: 
        arr: list
            list of element 
    Return: 
        list: return list arr list shuffled  
    """
    n = len(arr)
    for i in range(n-1):
        j = np.random.randint(i,n)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def lpa(G : nx.Graph, attribute : str, max_iter=100, verbose = False):
    """
    Label Propagation Algorithm

    Parameters:
        G: Networkx Graph 
            The graph on which the algorithm will be applied
        attribute: str
            The attribute of the nodes that will be used as labels
        max_iter: int (optional)
            Maximum number of iterations
        
    Return:
        Dict: where the key are the nodes id 
            and the values are the labels assiated to each node
    """
    labels = {n: G.nodes[n].get(attribute, None) for n in G.nodes}
    unique_labels = list(set(labels.values()))
    shuffled_ids = fisher_yates(list(G.nodes))
    if verbose:
        print(f"shuffled_ids: {shuffled_ids}")
    converged = False
    iter_ = 0
    while (not converged) and (iter_ < max_iter):
        prev_labels = labels.copy()

        for i, n in enumerate(G.nodes):
            # If the node has a label, we skip it
            if labels[n] is not None:
                continue
            neighbours = list(G.neighbors(n))
            # If the node has no neighbours, we give a random label
            if len(neighbours) == 0:
                labels[n] = np.random.choice(unique_labels)
                continue
            # Get the most common label among the neighbours
            neighbour_labels = [labels[nb] for nb in neighbours]
            most_common = max(set(neighbour_labels), key=neighbour_labels.count)
            labels[n] = most_common

        converged = prev_labels == labels
        iter_ += 1

    return labels

def remove_labels(G : nx.Graph, attribute : str, proportion : float):
    """
    Remove a certain proportion of labels from the graph

    Parameters:
        G: Networkx Graph 
            The graph on which the algorithm will be applied
        attribute: str
            The attribute of the nodes that will be used as labels
        proportion: float
            The proportion of labels to remove
        
    Return:
        Networkx Graph: The graph with the labels removed
    """
    Gcopy = G.copy()
    nodes = list(Gcopy.nodes)
    shuffled_ids = fisher_yates(nodes)
    nb_to_remove = len(nodes) * proportion
    for i, n in enumerate(shuffled_ids):
        if i < nb_to_remove:
            Gcopy.nodes[n][attribute] = None
    return Gcopy

def evaluate_lpa(G : nx.Graph, 
                 list_attr : list[str] = ["major_index", "dorm", "gender"],
                 list_prop : list[float] = [0.1, 0.2, 0.3, 0.4],
                 max_iter=100, verbose = False):
    """
    Evaluate the LPA algorithm on a graph

    Parameters:
        G: Networkx Graph 
            The graph on which the algorithm will be applied
        list_attr: list[str]
            List of attributes of the nodes that will be used as labels
        list_prop: list[float]
            List of proportions of labels to remove
        max_iter: int (optional)
            Maximum number of iterations

    Return:
        tuple(pd.DataFrame): A dataframe with the evaluation results,
            1st element: accuracy, 2nd element: mean absolute error, 3rd element: f1 score
    """
    results = []
    for attr in list_attr:
        for prop in list_prop:
            Gcopy = remove_labels(G, attr, prop)
            labels = lpa(Gcopy, attr, max_iter, verbose)
            results.append({
                "attribute": attr,
                "proportion": prop,
                "accuracy": sum([100 for n in Gcopy.nodes if Gcopy.nodes[n].get(attr, None) == labels[n]]) / len(Gcopy.nodes),
                "mean_absolute_error": sum([100 for n in Gcopy.nodes if Gcopy.nodes[n].get(attr, None) != labels[n]]) / len(Gcopy.nodes),
                "f1_score": sum([100 for n in Gcopy.nodes if Gcopy.nodes[n].get(attr, None) == labels[n]]) / len(Gcopy.nodes)
            })
    
    df_accuracy = pd.DataFrame(results).pivot(index="attribute", columns="proportion", values="accuracy")
    df_mae = pd.DataFrame(results).pivot(index="attribute", columns="proportion", values="mean_absolute_error")
    df_f1 = pd.DataFrame(results).pivot(index="attribute", columns="proportion", values="f1_score")

    return df_accuracy, df_mae, df_f1
    