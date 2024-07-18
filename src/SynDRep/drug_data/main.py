# -*- coding: utf-8 -*-

"""get drug data """

from itertools import combinations
from pathlib import Path

from tqdm import tqdm
from .physicochemical import get_physicochem_prop
from .graph_data import get_graph_data, generate_graph

import pandas as pd
import networkx as nx
import torch


def get_graph_and_physicochem_properties(
    kg_file: str | Path,
    kg_drug_file: str | Path,
    out_dir: str | Path,
    all_drug_prop_dict: dict = None,
    radius: int = 6,
    nBits: int = 2048,
    device="cuda",  # specify the device for calculations (cuda or cpu)
) -> pd.DataFrame:
    """
    This function generates a DataFrame containing physicochemical properties and graph data for all drug combinations.

    :param kg_file: The path to the knowledge graph file in TSV format.
    :param kg_drug_file: The path to the file containing drug names in CSV format.
    :param radius: The radius of the Morgan fingerprint. Defaults to 6.
    :param nBits: The number of bits in the fingerprint. Defaults to 2048.
    :param device: The device for calculations (cuda or cpu). Defaults to "cuda".

    :return: A DataFrame containing drug combinations, their physicochemical properties, and molecular fingerprints.
    """
    # get drug combinations
    drug_df = pd.read_csv(
        kg_drug_file,
    )
    drug_list = drug_df["Drug_name"].to_list()
    drug_combinations = list(combinations(drug_list, 2))

    Graph = generate_graph(kg_file=kg_file)

    pagerank_all = nx.pagerank(Graph)

    node_neighbors = {node: set(Graph.neighbors(node)) for node in Graph.nodes()}

    # Convert the neighborhood vectors to PyTorch tensors
    node_vectors = {
        node: torch.tensor(
            [1 if i in node_neighbors[node] else 0 for i in Graph.nodes()]
        )
        for node in tqdm(Graph.nodes())
    }

    data = []
    print(
        "Generating graph and physicochemical properties for all drug combinations..."
    )
    for drug_pair in tqdm(drug_combinations):
        
        drug1_name, drug2_name = drug_pair
        print('working on drug pair: ', drug1_name, '-', drug2_name, '...  ',flush=True)
        # get physicochemical properties
        ph_ch_df = get_physicochem_prop(
            drug1_name=drug1_name,
            drug2_name=drug2_name,
            all_drug_prop_dict=all_drug_prop_dict,
            radius=radius,
            nBits=nBits,
        )
        # get graph_data

        gr_df = get_graph_data(
            drug1_name=drug1_name,
            drug2_name=drug2_name,
            Graph=Graph,
            pagerank_all=pagerank_all,
            node_vectors=node_vectors,
            device=device,
        )
        
        gr_df = gr_df.drop(columns=["Drug1_name", "Drug2_name"], axis=1)
        if ph_ch_df is None:
            continue
        # combine the two dataframes
        row = pd.concat([ph_ch_df, gr_df], axis=1)
        data.append(row)

    # combine all dataframes into one
    df = pd.concat(data, axis=0)

    # save it to csv file
    Path(f"{out_dir}").mkdir(parents=True, exist_ok=True)
    df.to_csv(
        f"{out_dir}/all_drug_combinations_physicochemical_graph_data.csv", index=False
    )
    return df
