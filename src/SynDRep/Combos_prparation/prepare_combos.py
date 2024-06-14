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




def get_final_combinations(folder_path, output_path, Scoring_method= 'ZIP'):
    """_summary_

    Args:
        file_path (_type_): _description_
    """

    combos = merge_files(folder_path)

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


