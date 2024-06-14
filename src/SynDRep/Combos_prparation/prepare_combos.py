# -*- coding: utf-8 -*-

"""generate a unified combination values for each pair of drugs"""

import pathlib
import requests
from tqdm import tqdm
import pandas as pd
from statistics import mean
import json
from itertools import combinations
from pykeen.triples import TriplesFactory
import matplotlib.pyplot as plt
import numpy as np

name_cid_dict = {}


def prepare_combinations(
    combos_folder,
    kg_drug_file,
    in_kg_out_path="combinations_in_kg.csv",
    not_in_kg_out_path="combinations_not_in_kg.csv",
    results_txt_path="results.txt",
    Scoring_method="ZIP",
):

    # get the final combinations
    final_combos = get_final_combinations(combos_folder, Scoring_method)

    # load drug_df
    kg_drugs = pd.read_csv(kg_drug_file)

    # supply them with cids

    kg_drugs = add_cid((kg_drugs, "Drug_name"))

    # make a dictionary for drugs in KG
    kg_drug_CID = {}
    for i, cid in enumerate(kg_drugs["Drug_CID"]):

        if not pd.isna(cid):
            cid = int(cid)
            kg_drug_CID[cid] = kg_drugs["Drug_name"][i]
    n = final_combos.shape[0]
    final_combos["in_kg"] = None

    final_combos_drugs = set()

    for i, drug1_cid in enumerate(final_combos["Drug1_CID"]):
        drug1_cid = int(drug1_cid)
        drug2_cid = int(final_combos["Drug2_CID"][i])

        final_combos_drugs.add(drug1_cid)
        final_combos_drugs.add(drug2_cid)
        if kg_drug_CID.get(drug1_cid) and kg_drug_CID.get(drug2_cid):
            final_combos.loc[i, "in_kg"] = "yes"

    # remove combos not in kg
    not_in_kg = final_combos[final_combos["in_kg"].isna()]
    in_kg = final_combos.dropna(subset=["in_kg"])

    # make csv files
    in_kg.to_csv(in_kg_out_path, index=False)
    not_in_kg.to_csv(not_in_kg_out_path, index=False)

    # make text file with the results
    drug1_list = in_kg["Drug1_CID"].tolist()
    drug2_list = in_kg["Drug2_CID"].tolist()
    combos_kg_drugs = set(drug1_list + drug2_list)

    with open(results_txt_path, "w") as f:
        f.write(f"Total combinations: {n} \n")
        f.write(f"Total drugs in combinations: {len(final_combos_drugs)} \n")
        f.write(f"Total drugs in kg: {kg_drugs.shape[0]}\n")
        f.write(f"Total combinations in kg: {in_kg.shape[0]}\n")
        f.write(f"Total drugs have combinations in kg: {len(combos_kg_drugs)}\n")

    return final_combos


def get_final_combinations(folder_path, output_path, Scoring_method="ZIP"):
    """_summary_

    Args:
        file_path (_type_): _description_
    """

    combos = merge_files(folder_path)

    # supply them with cid

    combos = add_cid((combos, "Drug1_name"))
    combos = add_cid((combos, "Drug2_name"))

    # make CIDs as str
    combos[["Drug1_CID", "Drug2_CID"]] = combos[["Drug1_CID", "Drug2_CID"]].astype(str)

    # make scores as floats
    combos[["HSA", "Bliss", "Loewe", "ZIP"]] = combos[
        ["HSA", "Bliss", "Loewe", "ZIP"]
    ].astype(float)

    # group all combinations of the same 2 drugs together
    combos["DrugPair"] = combos[["Drug1_CID", "Drug2_CID"]].apply(
        lambda x: "-".join(sorted(x)), axis=1
    )
    combo_score = (
        combos.groupby(["DrugPair"])[["HSA", "Bliss", "Loewe", "ZIP"]]
        .agg(list)
        .reset_index()
    )
    # Apply the function to check if all values are positive, negative, or mixed
    combo_score["Value_Check"] = combo_score[Scoring_method].apply(check_pos_neg)

    # get the mean of values
    combo_score = combo_score.rename(
        columns={
            "HSA": "HSA_values",
            "Bliss": "Bliss_values",
            "Loewe": "Loewe_values",
            "ZIP": "ZIP_values",
        }
    )
    combo_score["HSA"] = combo_score["HSA_values"].apply(mean)
    combo_score["Bliss"] = combo_score["Bliss_values"].apply(mean)
    combo_score["Loewe"] = combo_score["Loewe_values"].apply(mean)
    combo_score["ZIP"] = combo_score["ZIP_values"].apply(mean)
    combo_score = combo_score.round(2)

    # get the cids again and remove mixed ZIPs
    combo_score[["Drug1_CID", "Drug2_CID"]] = (
        combo_score["DrugPair"].str.split("-", expand=True).dropna()
    )
    filtered_df = combo_score[~((combo_score["Value_Check"] == "Mixed"))]
    final_combos = filtered_df[
        [
            "Drug1_CID",
            "Drug2_CID",
            "HSA_values",
            "Bliss_values",
            "Loewe_values",
            "ZIP_values",
            "HSA",
            "Bliss",
            "Loewe",
            "ZIP",
        ]
    ].reset_index(drop=True)
    final_combos.to_csv(output_path, index=False)

    return final_combos


def merge_files(folder_path):
    """_summary_

    Args:
        folder_path (String): a path to the folder containing the combinations from databases in the recommended format

    Returns:
        dataframe: dataframe containing all the combinations together
    """

    # Get a list of dataframes of the files
    dfs = [pd.read_csv(f) for f in pathlib.Path(folder_path).iterdir() if f.is_file()]
    # concatenate all dfs
    all_combinations = pd.concat(dfs, ignore_index=True)
    return all_combinations


def check_pos_neg(lst):
    """Tells if all values are positives, negatives, zeros, or mixed

    Args:
        lst (list):list of values

    Returns:
        string: "All Positive", "All Negative", "Zero", or "Mixed"
    """
    all_positive = all(val > 0 for val in lst)
    all_negative = all(val < 0 for val in lst)
    zero = all(val == 0 for val in lst)
    return (
        "All Positive"
        if all_positive
        else ("All Negative" if all_negative else ("Zero" if zero else "Mixed"))
    )


def get_cid(drug_name):

    # add CID
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/txt"
    response = requests.get(url)
    if response.status_code == 200:
        cid = int(response.text.split("\n")[0])
    else:
        cid = None
    return cid


def add_cid(df, drug_name_column):
    """adds a column with drugs cid

    Args:
        df (dataframe): dataframe containing the name of drug
        drug_name_column (str): the title of drug name column
    """

    cid_column = drug_name_column.replace("name", "CID")
    df[cid_column] = None

    for i, name in tqdm(enumerate(df[drug_name_column])):
        cid = name_cid_dict.get(name)
        if cid:
            df.loc[i, cid_column] = cid
        else:
            cid = get_cid(name)
            name_cid_dict[name] = cid
    return df
