# -*- coding: utf-8 -*-

"""generate a unified combination values for each pair of drugs"""

import pathlib
import requests
from tqdm import tqdm
import pandas as pd
from statistics import mean
import json


def generate_enriched_kg(
    kg_file: str,
    combos_folder: str,
    kg_drug_file: str,
    out_dir: str,
    name_cid_dict: dict=None,
    scoring_method: str = "ZIP",
):
    """Produces an enriched KG with drug-drug combinations

    :param kg_file: a path to tsv file of KG.
    :param combos_folder: a path to the folder containing the drug combinations.
    :param kg_drug_file: a path to the csv file of KG drugs.
    :param out_dir: a path to the desired output directory.
    :param name_cid_dict: a dictionary of drug names to cid
    :param Scoring_method: a scoring method of the combination, defaults to "ZIP"
    :return: a df of enriched KG
    """
    
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    _, final_combos_with_relations = prepare_combinations(
        combos_folder, kg_drug_file, out_dir, name_cid_dict, scoring_method
    )
    combos_df = final_combos_with_relations[["Source_label", ":TYPE", "Target_label"]]
    kg_df = pd.read_table(
        kg_file, header=None, names=["Source_label", ":TYPE", "Target_label"]
    )

    enriched_kg = pd.concat([combos_df, kg_df], ignore_index=False)
    enriched_kg.to_csv(
        f"{out_dir}/enriched_kg.tsv", sep="\t", index=False, header=False
    )
    return enriched_kg


def prepare_combinations(
    combos_folder: str,
    kg_drug_file: str,
    out_dir: str,
    name_cid_dict: dict=None,
    scoring_method: str = "ZIP",
):
    """Prepare the drug combinations and produce the ones that can be added to KG

    :param combos_folder: a path to the folder containing the drug combinations.
    :param kg_drug_file: a path to the csv file of KG drugs.
    :param out_dir: a path to the desired output directory.
    :param name_cid_dict: a dictionary of drug names to cid, defaults to {}
    :param Scoring_method: a scoring method of the combination, defaults to "ZIP"
    :return: a final combinations to be added to KG
    """

    # load drug_df
    kg_drugs = pd.read_csv(kg_drug_file)

    # supply them with cids

    name_cid_dict, kg_drugs = add_cid(kg_drugs, "Drug_name", name_cid_dict)
    kg_drugs = kg_drugs.dropna(subset=["Drug_CID"])

    # export the name_cid_dict
    json.dump(name_cid_dict, open(f"{out_dir}/name_cid_dict.json", "w"), indent=4)

    # get the final combinations
    final_combos = get_merged_combinations(
        folder_path=combos_folder,
        name_cid_dict=name_cid_dict,
        output_path=out_dir,
        scoring_method=scoring_method,
    )

    # make a dictionary for drugs in KG
    kg_drug_CID = {}
    for i, row in kg_drugs.iterrows():
        cid = row["Drug_CID"]
        kg_drug_CID[cid] = row["Drug_name"]

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
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    in_kg.to_csv(f"{out_dir}/combinations_in_kg.csv", index=False)
    not_in_kg.to_csv(f"{out_dir}/combinations_not_in_kg.csv", index=False)

    # make text file with the results
    drug1_list = in_kg["Drug1_CID"].tolist()
    drug2_list = in_kg["Drug2_CID"].tolist()
    combos_kg_drugs = set(drug1_list + drug2_list)
    in_kg[":TYPE"] = in_kg[scoring_method].apply(synergy_detection)

    with open(f"{out_dir}/results.txt", "w") as f:
        f.write(f"Total combinations: {n} \n")
        f.write(f"Total drugs in combinations: {len(final_combos_drugs)} \n")
        f.write(f"Total drugs in kg: {kg_drugs.shape[0]}\n")
        f.write(f"Total combinations to be added to kg: {in_kg.shape[0]}\n")
        f.write(f"Total drugs have combinations in kg: {len(combos_kg_drugs)}\n")
        f.write(
            f"Total HAS_SYNERGISM_WITH combinations: {len(in_kg[in_kg[':TYPE']=='HAS_SYNERGISM_WITH'])}\n"
        )
        f.write(
            f"Total HAS_ANTAGONISM_WITH combinations: {len(in_kg[in_kg[':TYPE']=='HAS_ANTAGONISM_WITH'])}\n"
        )
        f.write(
            f"Total HAS_ADDITIVE_EFFECT_WITH combinations: {len(in_kg[in_kg[':TYPE']=='HAS_ADDITIVE_EFFECT_WITH'])}\n"
        )

    relations = []

    # Iterate through the input DataFrame
    for i, row in tqdm(in_kg.iterrows(), total=len(in_kg)):
        drug1, drug2 = row["Drug1_name"], row["Drug2_name"]
        drug1_cid, drug2_cid = row["Drug1_CID"], row["Drug2_CID"]
        hsa, bliss, loewe, zip, type = (
            row["HSA"],
            row["Bliss"],
            row["Loewe"],
            row["ZIP"],
            row[":TYPE"],
        )

        # Append both directions of the relationships
        relations.append(
            [drug1, drug2, drug1_cid, drug2_cid, hsa, bliss, loewe, zip, type]
        )
        relations.append(
            [drug2, drug1, drug2_cid, drug1_cid, hsa, bliss, loewe, zip, type]
        )

    # Create a DataFrame from the list and remove duplicates
    columns = [
        "Source_label",
        "Target_label",
        "Source_CID",
        "Target_CID",
        "HSA",
        "Bliss",
        "Loewe",
        "ZIP",
        ":TYPE",
    ]
    has_comb = pd.DataFrame(relations, columns=columns).drop_duplicates()

    # Remove self-interactions
    has_comb = has_comb[has_comb["Source_label"] != has_comb["Target_label"]]
    final_combos_with_relations = has_comb[
        [
            "Source_label",
            "Source_CID",
            ":TYPE",
            "Target_label",
            "Target_CID",
            "HSA",
            "Bliss",
            "Loewe",
            "ZIP",
        ]
    ]

    # Save to CSV
    final_combos_with_relations.to_csv(
        f"{out_dir}/has_combination_with.csv", index=False
    )

    return final_combos, final_combos_with_relations


def get_merged_combinations(
    folder_path, output_path, name_cid_dict, scoring_method="ZIP"
):
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
    combo_score["Value_Check"] = combo_score[scoring_method].apply(check_pos_neg)

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

    # get names

    cid_name_dict = {str(v): k for k, v in name_cid_dict.items()}

    filtered_df["Drug1_name"] = filtered_df["Drug1_CID"].apply(cid_name_dict.get)
    filtered_df["Drug2_name"] = filtered_df["Drug2_CID"].apply(cid_name_dict.get)

    final_combos = filtered_df[
        [
            "Drug1_CID",
            "Drug1_name",
            "Drug2_CID",
            "Drug2_name",
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
    final_combos.to_csv(f"{output_path}/all_combos.csv", index=False)

    return final_combos


def merge_files(folder_path):
    """concatenates all csv files  in the directory

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
    """get the PubChem ID for a drug

    Args:
        drug_name (str): drug name

    Returns:
        int: the cid of a drug
    """

    # add CID
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/cids/txt"
    )
    response = requests.get(url)
    if response.status_code == 200:
        cid = int(response.text.split("\n")[0])
    else:
        cid = None
    return cid


def add_cid(df, drug_name_column, name_cid_dict=None):
    """adds a column with drugs cid

    Args:
        df (dataframe): dataframe containing the name of drug
        drug_name_column (str): the title of drug name column
    """

    cid_column = drug_name_column.replace("name", "CID")
    df[cid_column] = None

    for i, row in tqdm(df.iterrows(), total=len(df)):
        name = row[drug_name_column]
        
        if name_cid_dict:
            cid = name_cid_dict.get(name)
            if cid:
                df.loc[i, cid_column] = cid
            else:
                cid = get_cid(name)
                name_cid_dict[name] = cid
                df.loc[i, cid_column] = cid
        else:
            name_cid_dict={}
            cid = get_cid(name)
            name_cid_dict[name] = cid
            df.loc[i, cid_column] = cid
    return name_cid_dict, df


def synergy_detection(value):
    """_summary_

    Args:
        value (_type_): _description_

    Returns:
        str: combination type as symergism, antgonism, or addition
    """
    if value > 0:
        return "HAS_SYNERGISM_WITH"
    elif value < 0:
        return "HAS_ANTAGONISM_WITH"
    else:
        return "HAS_ADDITIVE_EFFECT_WITH"
