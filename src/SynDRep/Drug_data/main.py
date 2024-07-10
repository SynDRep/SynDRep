# -*- coding: utf-8 -*-

"""get drug data """

from itertools import combinations
from pathlib import Path
from physichchemical import  get_physicochem_prop
from graph_data import get_graph_data, generate_graph

import pandas as pd


def get_graph_and_physicochem_properties(
    kg_file: str | Path,
    kg_drug_file:str | Path,
    radius: int = 6,
    nBits: int = 2048,
    device="cuda"  # specify the device for calculations (cuda or cpu)
)->pd.DataFrame:
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

    
    data = []
    for drug_pair in drug_combinations:
        drug1_name, drug2_name = drug_pair
        # get physicochemical properties
        ph_ch_df = get_physicochem_prop(drug1_name, drug2_name, radius=radius, nBits=nBits)
        #get graph_data
        Graph = generate_graph(kg_file=kg_file)
        gr_df = get_graph_data(drug1_name, drug2_name, Graph, device)
        
        # combine the two dataframes
        row = pd.concat([ph_ch_df, gr_df], axis=1)
        data.append(row)
    
    # combine all dataframes into one
    df = pd.concat(data, axis=0)
    return df
        