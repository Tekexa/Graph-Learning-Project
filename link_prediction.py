from abc import ABC
from abc import abstractmethod
import networkx as nx
import numpy as np
from tqdm import trange
import pandas as pd

class LinkPrediction(ABC):
    def __init__(self, graph, verbose = False):
        """
        Constructor

        Parameters
        ----------
        graph : Networkx graph
        """
        self.graph = graph
        self.N = len(graph)
        self.score = np.zeros((self.N, self.N))
        self.part_graph = graph  # at init, the graph is not modified
        self.verbose = verbose

    def neighbors(self, v, mode = "graph"):
        """
        Return the neighbors list of a node

        Parameters
        ----------
        v : int
            node id
        mode : string ("graph" or "part_graph")
            use the original graph or the graph with proportion of edges removed

        Return
        ------
        neighbors_list : python list
        """
        v = str(v)
        if mode == "graph":
            neighbors_list = self.graph.neighbors(v)
        elif mode == "part_graph":
            neighbors_list = self.part_graph.neighbors(v)
        else:
            raise ValueError("mode must be 'graph' or 'part_graph'")
        return list(neighbors_list)

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Fit must be implemented")
    
    def remove_prop_edges(self, proportion):
        """
        Remove a proportion of edges from the graph

        Parameters
        ----------
        proportion : float
            proportion of edges to remove

        Return
        ------
        graph : Networkx graph
            graph with proportion of edges removed
        """
        graph = self.graph.copy()
        edges = list(graph.edges())
        edges_to_remove = np.random.choice(len(edges), int(proportion*len(edges)), replace=False)
        for i in edges_to_remove:
            graph.remove_edge(*edges[i])
        self.part_graph = graph
    
    def predict_k_edges(self, k):
        """
        Predict k edges that are most likely to appear

        Parameters
        ----------
        k : int
            number of edges to predict

        Return
        ------
        edges : list of tuples
            list of predicted edges
        """
        prev_score = self.score.copy()
        edges = []
        if self.verbose:
            range_k = trange(k)
        else:
            range_k = range(k)
        for i in range_k:
            idx = np.unravel_index(np.argmax(self.score, axis=None), self.score.shape)
            self.score[idx] = 0
            self.score[idx[1], idx[0]] = 0  # to avoid duplicates
            edges.append(idx)

        self.score = prev_score
        return edges
    
    def evaluate(self, list_proportions = [0.05, 0.1, 0.15, 0.2],
                 list_k_s = [50, 100, 200, 400]):
        """
        Evaluate the model

        Parameters
        ----------
        list_proportions : list of float
            list of proportion of edges to remove
        list_k_s : list of int
            list of number of edges to predict

        Return
        ------
        scores : tuple of pd.DataFrame
            tuple of pd.DataFrame for top_scores, precision_scores and recall_scores
        """
        # show_edges = True
        top_scores = np.zeros((len(list_proportions), len(list_k_s)))
        precision_scores = np.zeros((len(list_proportions), len(list_k_s)))
        recall_scores = np.zeros((len(list_proportions), len(list_k_s)))
        
        # Save the true edges of the graph when it is complete
        true_edges = list(self.graph.edges())
        true_edges = set([(min(i,j), max(i,j)) for i,j in true_edges])

        for proportion_ in list_proportions:
            self.remove_prop_edges(proportion_)
            if self.verbose:
                print("\n=====================================")
                print(f"Remaining edges: {len(self.part_graph.edges())} for prop.: {proportion_}")
            for k_ in list_k_s:
                # Update the score matrix
                self.fit()
                pred_edges = self.predict_k_edges(k_)
                # if show_edges:
                #     print(f"pred_edges: {pred_edges}, len(pred_edges): {len(pred_edges)}")
                #     show_edges = False
                # if show_edges:
                #     print(f"len(true_edges): {len(true_edges)}, true_edges: {true_edges}")
                #     show_edges = False
                pred_edges = set([(str(min(i,j)), str(max(i,j))) for i,j in pred_edges])
                # if show_edges:
                #     print(f"len(pred_edges): {len(pred_edges)}, pred_edges: {pred_edges}")
                #     show_edges = False
                if self.verbose:
                    print(f"Nb of predicted edges: {len(pred_edges)}")

                true_positives = len(true_edges.intersection(pred_edges))
                false_positives = len(pred_edges.difference(true_edges))
                false_negatives = len(true_edges.difference(pred_edges))

                if self.verbose:
                    print(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")

                if true_positives == 0:
                        continue
                
                curr_prop = list_proportions.index(proportion_)
                curr_k = list_k_s.index(k_)

                top_scores[curr_prop, curr_k] = 100 * float(true_positives)/k_
                precision_scores[curr_prop, curr_k] = 100 * float(true_positives)/(true_positives + false_positives)
                recall_scores[curr_prop, curr_k] = 100 * float(true_positives)/(true_positives + false_negatives)

                if self.verbose:
                    print(f"Prop.: {proportion_}, k: {k_} (pos: ({curr_prop}, {curr_k})), top: {top_scores[curr_prop, curr_k]}, prec: {precision_scores[curr_prop, curr_k]}, rec: {recall_scores[curr_prop, curr_k]}")
        
        # Create pd.DataFrame for the results
        df_top = pd.DataFrame(top_scores, index=list_proportions, columns=list_k_s, dtype=float)
        df_precision = pd.DataFrame(precision_scores, index=list_proportions, columns=list_k_s, dtype=float)
        df_recall = pd.DataFrame(recall_scores, index=list_proportions, columns=list_k_s, dtype=float)

        print("Done")
        return df_top, df_precision, df_recall
    

class CommonNeighbors(LinkPrediction):
    def __init__(self, graph, verbose = False):
        super(CommonNeighbors, self).__init__(graph, verbose)

    def fit(self):
        """
        Compute the Common Neighbors index

        Return
        ------
        score : numpy array
            score[i,j] is the number of common neighbors between i and j
        """
        score = np.zeros((self.N, self.N))
        if self.verbose:
            range_N = trange(self.N)
        else:
            range_N = range(self.N)
        for i in range_N:
            for j in range(i+1, self.N):
                neighbors_i = set(self.neighbors(i, mode="part_graph"))
                neighbors_j = set(self.neighbors(j, mode="part_graph"))
                common = len(neighbors_i.intersection(neighbors_j))
                score[i,j] = common
                score[j,i] = common
        self.score = score

        return score
    
class Jaccard(LinkPrediction):
    def __init__(self, graph, verbose = False):
        super(Jaccard, self).__init__(graph, verbose)

    def fit(self):
        """
        Compute the Jaccard index

        Return
        ------
        score : numpy array
            score[i,j] is the Jaccard index between i and j
        """
        score = np.zeros((self.N, self.N))
        if self.verbose:
            range_N = trange(self.N)
        else:
            range_N = range(self.N)
        for i in range_N:
            for j in range(i+1, self.N):
                neighbors_i = set(self.neighbors(i, mode="part_graph"))
                neighbors_j = set(self.neighbors(j, mode="part_graph"))
                if len(neighbors_i.union(neighbors_j)) == 0:
                    score[i,j] = 0
                    score[j,i] = 0
                else:
                    score[i,j] = len(neighbors_i.intersection(neighbors_j))/len(neighbors_i.union(neighbors_j))
                    score[j,i] = score[i,j]
        self.score = score

class AdamicAdar(LinkPrediction):
    def __init__(self, graph, verbose = False):
        super(AdamicAdar, self).__init__(graph, verbose)

    def fit(self):
        """
        Compute the Adamic Adar index

        Return
        ------
        score : numpy array
            score[i,j] is the Adamic Adar index between i and j
        """
        score = np.zeros((self.N, self.N))
        if self.verbose:
            range_N = trange(self.N)
        else:
            range_N = range(self.N)
        for i in range_N:
            for j in range(i+1, self.N):
                neighbors_i = set(self.neighbors(i, mode="part_graph"))
                neighbors_j = set(self.neighbors(j, mode="part_graph"))
                common_neighbors = neighbors_i.intersection(neighbors_j)
                if len(common_neighbors) == 0:
                    score[i,j] = 0
                    score[j,i] = 0
                else:
                    score[i,j] = np.sum([1/np.log(len(self.neighbors(v))) for v in common_neighbors])
                    score[j,i] = score[i,j]
        self.score = score
    




# =========================================================================
# ============================== USELESS ==================================
# =========================================================================

class AggregateScores(LinkPrediction):
    def __init__(self, graph, scores):
        """
        Constructor

        Parameters
        ----------
        graph : Networkx graph
        scores : list of numpy array
            list of scores to aggregate

        This class could be used to evaluate the performance of the aggregation
        of different predictors
        """
        super(AggregateScores, self).__init__(graph)
        self.scores = scores

    def fit(self):
        """
        Compute the Aggregate score

        Return
        ------
        score : numpy array
            score[i,j] is the Aggregate score between i and j
        """
        score = np.zeros((self.N, self.N))
        for s in self.scores:
            score += s
        return score
    
    def predict_k_edges(self, k):
        """
        Predict k edges that are most likely to appear

        Parameters
        ----------
        k : int
            number of edges to predict

        Return
        ------
        edges : list of tuples
            list of predicted edges
        """
        score = self.fit()
        edges = []
        for i in trange(k):
            idx = np.unravel_index(np.argmax(score, axis=None), score.shape)
            edges.append(idx)
            score[idx] = 0
        return edges
    
    def remove_edges(self, og_graph, proportion):
        """
        Remove a proportion of edges from the graph

        Parameters
        ----------
        og_graph : Networkx graph
            original graph
        proportion : float
            proportion of edges to remove

        Return
        ------
        graph : Networkx graph
            graph with proportion of edges removed
        """
        graph = og_graph.copy()
        edges = list(graph.edges())
        edges_to_remove = np.random.choice(len(edges), int(proportion*len(edges)), replace=False)
        for i in edges_to_remove:
            graph.remove_edge(*edges[i])
        return graph
    
    def evaluate(self, list_proportions, list_k_s):
        """
        Evaluate the model

        Parameters
        ----------
        list_proportions : list of float
            list of proportion of edges to remove
        list_k_s : list of int
            list of number of edges to predict

        Return
        ------
        scores : list of float
            list of scores
        """
        top_scores = np.zeros((len(list_proportions), len(list_k_s)))
        precision_scores = np.zeros((len(list_proportions), len(list_k_s)))
        recall_scores = np.zeros((len(list_proportions), len(list_k_s)))
        for proportion in list_proportions:
            part_graph = self.remove_edges(self.graph, proportion)
            for k in list_k_s:
                pred_edges = self.predict_k_edges(k)
                true_edges = list(self.graph.edges())
                true_edges = set([(min(i,j), max(i,j)) for i,j in true_edges])
                pred_edges = set([(min(i,j), max(i,j)) for i,j in pred_edges])

                true_positives = len(true_edges.intersection(pred_edges))
                false_positives = len(pred_edges.difference(true_edges))
                false_negatives = len(true_edges.difference(pred_edges))

                curr_prop = list_proportions.index(proportion)
                curr_k = list_k_s.index(k)

                top_scores[curr_prop, curr_k] = true_positives/k
                precision_scores[curr_prop, curr_k] = true_positives/(true_positives + false_positives)
                recall_scores[curr_prop, curr_k] = true_positives/(true_positives + false_negatives)
