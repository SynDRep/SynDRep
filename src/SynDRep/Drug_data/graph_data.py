# -*- coding: utf-8 -*-

"""get drug data from KG"""

from pathlib import Path
import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm


def get_graph_data(
    drug1_name: str,
    drug2_name: str,
    Graph: nx.DiGraph,
    pagerank_all: dict=None,
    node_vectors: dict=None,
    device: str = 'cuda',
)-> pd.DataFrame:
    """
    This function gets graph data for two drugs.
    
    :param drug1_name: The name of the first drug.
    :param drug2_name: The name of the second drug.
    :param Graph: The directed graph to analyze.
    :param device: The device to use for computations. Defaults to 'cuda'.
    
    :return: A dataframe containing the graph data for the two drugs.
    """

    row = {}
    row["Drug1_name"] = drug1_name
    row["Drug2_name"] = drug2_name
    row['Drug1_degree'] = get_drug_node_degree(Graph,drug1_name)
    row['Drug2_degree'] = get_drug_node_degree(Graph, drug2_name)
    row['Drug1_clustering_coefficient'] = get_drug_clustering_coefficient(Graph,drug1_name)
    row['Drug2_clustering_coefficient'] = get_drug_clustering_coefficient(Graph, drug2_name)
    row['Drug1_pagerank'] = get_drug_page_rank(Graph, drug1_name, pagerank_all)
    row['Drug2_pagerank'] = get_drug_page_rank(Graph, drug2_name, pagerank_all)
    row['Shortest_path_length'] = get_shortest_path_length(Graph, drug1_name, drug2_name)
    row['Cosine_similarity'] = get_cosine_similarity(Graph,drug1_name,drug2_name, node_vectors, device)
    return pd.DataFrame(row,index=[0])
    
def generate_graph(
    kg_file: str | Path,
) -> nx.DiGraph:
    
    # Step 1: Read Data from TSV
    
    with open(kg_file, 'r') as f:
        lines = f.readlines()

    # Step 2: Create Graph and Add Edges
    G = nx.DiGraph()  # You can choose a different graph type if needed

    for line in lines:
        source, relation, target= line.strip().split('\t')  # Assuming the format is source<TAB>target<TAB>weight
        G.add_edge(source, target, label= relation)  
    return G

def get_drug_node_degree(Graph: nx.DiGraph, drug_name: str) -> int:
    """
    This function retrieves the degree of a drug node in the graph.

    :param Graph: The directed graph to analyze.
    :param drug_name: The name of the drug.
    
    :return: The degree of the drug node in the graph.
    """

    return Graph.degree(drug_name)


def get_shortest_path_length(Graph: nx.DiGraph, drug1: str, drug2: str) -> int:
    """
    This function retrieves the shortest path length between two drug nodes in the graph.

    :param Graph: The directed graph to analyze.
    :param drug1: The name of the first drug.
    :param drug2: The name of the second drug.
    
    :return: The shortest path length between the two drug nodes.
    """

    shortest_path_length = (
        nx.shortest_path_length(Graph, drug1, drug2)
        if nx.has_path(Graph, drug1, drug2)
        else -1
    )
    return shortest_path_length


def get_drug_clustering_coefficient(Graph: nx.DiGraph, drug_name: str) -> float:
    """
    This function retrieves the clustering coefficient of a drug node in the graph.

    :param Graph: The directed graph to analyze.
    :param drug_name: The name of the drug.
    
    :return: The clustering coefficient of the drug node in the graph.
    """
    
    return nx.clustering(Graph, drug_name)


def get_drug_page_rank(Graph: nx.DiGraph, drug_name: str, pagerank_all: dict=None) -> float:
    """
    This function retrieves the PageRank of a drug node in the graph.

    :param Graph: The directed graph to analyze.
    :param drug_name: The name of the drug.
    :param pagerank_all: The dictionary containing the PageRank values of all nodes. Defaults to None
    
    :return: The PageRank of the drug node in the graph.
    """
    if pagerank_all is None:
        pagerank_all = nx.pagerank(Graph)
    return pagerank_all.get(drug_name)

def get_cosine_similarity(
    Graph: nx.DiGraph, drug1_name: str, drug2_name: str, node_vectors: dict=None, device="cuda"
) -> float:
    """
    This function calculates the cosine similarity between two drug nodes in the graph.

    :param Graph: The directed graph to analyze.
    :param drug1: The name of the first drug.
    :param drug2: The name of the second drug.
    :param device: The device to use for calculations. Defaults to "cuda".
    
    :return: The cosine similarity between the two drug nodes in the graph.
    """
    
    if node_vectors is None:
        node_neighbors = {node: set(Graph.neighbors(node)) for node in Graph.nodes()}

        # Convert the neighborhood vectors to PyTorch tensors
        node_vectors = {
            node: torch.tensor(
                [1 if i in node_neighbors[node] else 0 for i in Graph.nodes()]
            )
            for node in tqdm(Graph.nodes())
        }
    

    # Get the neighborhoods of the nodes

    vector_drug1 = (
        node_vectors[drug1_name].float().cuda()
        if device == "cuda"
        else node_vectors[drug1_name].float()
    )
    vector_drug2 = (
        node_vectors[drug2_name].float().cuda()
        if device == "cuda"
        else node_vectors[drug2_name].float()
    )

    # Calculate cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        vector_drug1.view(1, -1), vector_drug2.view(1, -1)
    )

    return cosine_sim.item()
