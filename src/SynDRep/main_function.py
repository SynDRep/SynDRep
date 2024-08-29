# -*- coding: utf-8 -*-
"""main SynDRep function"""
import itertools
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from itertools import combinations
from SynDRep.combos_preparation import generate_enriched_kg
from SynDRep.drug_data import get_graph_and_physicochem_properties, get_physicochem_prop
from SynDRep.embedding import compare_embeddings, embed_and_predict
from SynDRep.ML import classify_data


def run_SynDRep(
    combos_folder: str | Path,
    drug_class_name: str,
    em_models_names: List[str],
    kg_drug_file: str | Path,
    kg_file: str | Path,
    kg_labels_file: str | Path,
    ml_model_names: List[str],
    out_dir: str | Path,
    optimizer_name: str,
    scoring_metrics: List[str],
    validation_cv: int,
    all_drug_drug_predictions: bool = False,
    all_drug_prop_dict: dict = None,
    all_out_file: str = None,
    best_out_file: str = "predictions_best.csv",
    config_path: str | Path = None,
    device: str = "cuda",
    filter_training: bool = False,
    method: str = "Embedding_only",
    nBits: int = 2048,
    name_cid_dict: dict = None,
    pred_reverse: bool = False,
    predict_all: bool = False,
    radius: int = 6,
    rand_labels: bool = False,
    scoring_method: str = "ZIP",
    sorted_predictions: bool = True,
    subsplits: bool = True,
) -> pd.DataFrame:
    """
    Main function of SynDRep.

    :param combos_folder: the folder containing the drug combinations.
    :param drug_class_name: the string indicating the drug class in the KG.
    :param em_models_names: the names of the models to use for embedding.
    :param kg_drug_file: the path to the csv file containing drug names.
    :param kg_file: the path to the original KG file.
    :param kg_labels_file: the path to the tsv file containing KG labels.
    :param ml_model_names: the names of the models to use for ML classification.
    :param out_dir: the output directory.
    :param optimizer_name: the name of the optimizer.
    :param scoring_metrics: the scoring metrics to use for ML-models scoring.
    :param validation_cv: the number of cross-validation folds for ML-model tuning.
    :param all_drug_drug_predictions: whether to save all drug-drug combinations predictions. Defaults to False.
    :param all_drug_prop_dict: a dictionary containing physicochemical properties for all drugs. Defaults to None.
    :param all_out_file: the output file for all embedding predictions. Defaults to None.
    :param best_out_file: the output file for the best embedding predictions. Defaults to "predictions_best.csv".
    :param config_path: the path to the embedding model configuration file. Defaults to None.
    :param device: the device to use for computations. Defaults to "cuda".
    :param filter_training: whether to filter the training data from relations predicted by embedding. Defaults to False.
    :param method: Method to use for syndrep. Please choose from: Embedding_only, Embedding_then_ML, Data_extraction_then_ML, physicochemical_data_and_embedding_then_ML". Defaults to "Embedding_only".
    :param nBits: the number of bits for Morgen fingerprint calculation. Defaults to 2048.
    :param name_cid_dict: a dictionary containing drug names and CIDs. Defaults to None.
    :param pred_reverse: whether to predict reverse relations using embedding models. Defaults to False.
    :param predict_all: whether to save all test drug-drug relations predicted by Embedding models . Defaults to False.
    :param radius: the radius for Morgen fingerprint calculation. Defaults to 6.
    :param rand_labels: whether to randomize the labels for ML classification. Defaults to False.
    :param scoring_method: the synergy scoring method to use. Defaults to "ZIP".
    :param sorted_predictions: whether to sort the predictions (in embedding_only method). Defaults to True.
    :param subsplits: whether to use subsplits for training of embedding models. Defaults to True.

    :return: a DataFrame containing the best predicted relation for each pair drugs.
    """

    if method == "Embedding_only":
        generate_enriched_kg(
            combos_folder=combos_folder,
            kg_drug_file=kg_drug_file,
            kg_file=kg_file,
            name_cid_dict=name_cid_dict,
            out_dir=out_dir,
            scoring_method=scoring_method,
        )

        prediction_all, prediction_best = embed_and_predict(
            all_drug_drug_predictions=all_drug_drug_predictions,
            all_out_file=all_out_file,
            best_out_file=best_out_file,
            config_path=config_path,
            drug_class_name=drug_class_name,
            filter_training=filter_training,
            get_embeddings_data=False,
            kg_drug_file=kg_drug_file,
            kg_file=f"{out_dir}/enriched_kg.tsv",
            kg_labels_file=kg_labels_file,
            models_names=em_models_names,
            out_dir=out_dir,
            pred_reverse=pred_reverse,
            predict_all_test=predict_all,
            sorted_predictions=sorted_predictions,
            subsplits=subsplits,
        )
        return prediction_best

    elif method == "Embedding_then_ML":
        if name_cid_dict is None:
            # load drug_df
            kg_drugs = pd.read_csv(kg_drug_file)

            # get name_cid_dict
            name_cid_dict, kg_drugs = add_cid(kg_drugs, "Drug_name", name_cid_dict)
        # get embeddings
        best_model = compare_embeddings(
            all_out_file=all_out_file,
            best_out_file=best_out_file,
            config_path=config_path,
            drug_class_name=drug_class_name,
            filter_training=filter_training,
            get_embeddings_data=True,
            kg_file=kg_file,
            kg_labels_file=kg_labels_file,
            models_names=em_models_names,
            out_dir=out_dir,
            predict_all=predict_all,
            subsplits=subsplits,
        )
        embeddings = pd.read_csv(
            f"{out_dir}/{best_model}/{best_model}_embeddings_drugs.csv"
        )
        # Get the list of embedding columns
        embedding_columns = [
            col for col in embeddings.columns if col.startswith("embedding_")
        ]

        # Create an empty DataFrame for the combined embeddings
        combined_rows = []
        drug_combinations = list(combinations(embeddings.iterrows(), 2))
        # Iterate through all combinations of node pairs
        for (idx1, row1), (idx2, row2) in tqdm(drug_combinations):
            combined_row = {}

            # Add the node ids to the combined row
            combined_row["Drug1_name"] = row1["entity"]
            combined_row["Drug2_name"] = row2["entity"]
            
            # Add the embeddings to the combined row
            for col in embedding_columns:
                combined_row[f"Drug1_{col}"] = row1[col]
                combined_row[f"Drug2_{col}"] = row2[col]
            combined_rows.append(combined_row)

        combined_df = pd.DataFrame(combined_rows)

        combined_df.to_csv(
            f"{best_model}_drug_combinations_embeddings.csv", index=False
        )

        input_columns = [col for col in combined_df.columns if "embedding_" in col]
        
        generate_enriched_kg(
        combos_folder = combos_folder,
        kg_drug_file=kg_drug_file,
        kg_file=kg_file,
        name_cid_dict=name_cid_dict,
        out_dir=out_dir,
        scoring_method=scoring_method,
        )
        
        data_for_training, data_for_prediction = get_ML_train_and_prediction_data(
            all_input_df=combined_df,
            name_cid_dict=name_cid_dict,
            out_dir=out_dir,
            input_columns=input_columns,
            scoring_method=scoring_method,
        )

        predicted_df = classify_data(
            data_for_prediction=data_for_prediction,
            data_for_training=data_for_training,
            model_names=ml_model_names,
            optimizer_name=optimizer_name,
            out_dir=out_dir,
            rand_labels=rand_labels,
            scoring_metrics=scoring_metrics,
            validation_cv=validation_cv,
        )
        return predicted_df

    elif method == "Data_extraction_then_ML":

        combined_df = get_graph_and_physicochem_properties(
            kg_file=kg_file,
            kg_drug_file=kg_drug_file,
            out_dir=out_dir,
            all_drug_prop_dict=all_drug_prop_dict,
            device=device,
            radius=radius,
            nBits=nBits,
        )

        # remove CIDs
        combined_df = combined_df.drop(['Drug1_CID', 'Drug2_CID'], axis=1)

        input_columns = [
            col
            for col in combined_df.columns
            if ("name" not in col) and ("CID" not in col)
        ]
        
        generate_enriched_kg(
        combos_folder = combos_folder,
        kg_drug_file=kg_drug_file,
        kg_file=kg_file,
        name_cid_dict=name_cid_dict,
        out_dir=out_dir,
        scoring_method=scoring_method,
        )
        

        data_for_training, data_for_prediction = get_ML_train_and_prediction_data(
            all_input_df=combined_df,
            name_cid_dict=name_cid_dict,
            out_dir=out_dir,
            input_columns=input_columns,
            scoring_method=scoring_method,
        )

        predicted_df = classify_data(
            data_for_prediction=data_for_prediction,
            data_for_training=data_for_training,
            model_names=ml_model_names,
            optimizer_name=optimizer_name,
            out_dir=out_dir,
            rand_labels=rand_labels,
            scoring_metrics=scoring_metrics,
            validation_cv=validation_cv,
        )
        return predicted_df

    elif method == "physicochemical_data_and_embedding_then_ML":
        # get embeddings
        best_model = compare_embeddings(
            all_out_file=all_out_file,
            best_out_file=best_out_file,
            config_path=config_path,
            drug_class_name=drug_class_name,
            filter_training=filter_training,
            get_embeddings_data=True,
            kg_file=kg_file,
            kg_labels_file=kg_labels_file,
            models_names=em_models_names,
            out_dir=out_dir,
            predict_all=predict_all,
            subsplits=subsplits,
        )
        embeddings = pd.read_csv(
            f"{out_dir}/{best_model}/{best_model}_embeddings_drugs.csv"
        )
        # Get the list of embedding columns
        embedding_columns = [
            col for col in embeddings.columns if col.startswith("embedding_")
        ]

        # Create an empty DataFrame for the combined embeddings
        combined_rows = []
        drug_combinations = list(combinations(embeddings.iterrows(), 2))
        # Iterate through all combinations of node pairs
        for (idx1, row1), (idx2, row2) in tqdm(drug_combinations):
            embedding_row = {}

            # get physicochemical data
            combined_row = get_physicochem_prop(
                drug1_name=row1["entity"],
                drug2_name=row2["entity"],
                all_drug_prop_dict=all_drug_prop_dict,
                radius=radius,
                nBits=nBits,
            )

            # Add the embeddings to the combined row
            for col in embedding_columns:
                combined_row[f"Drug1_{col}"] = row1[col]
                combined_row[f"Drug2_{col}"] = row2[col]
            combined_rows.append(combined_row)

        combined_df = pd.concat(combined_rows).reset_index(inplace=True, drop=True)

        combined_df.to_csv(
            f"{best_model}_drug_combinations_physicochemical_and_embeddings.csv",
            index=False,
        )
        
        # remove CIDs
        combined_df = combined_df.drop(['Drug1_CID', 'Drug2_CID'], axis=1)

        input_columns = [
            col
            for col in combined_df.columns
            if ("name" not in col) and ("CID" not in col)
        ]
        
        generate_enriched_kg(
        combos_folder = combos_folder,
        kg_drug_file=kg_drug_file,
        kg_file=kg_file,
        name_cid_dict=name_cid_dict,
        out_dir=out_dir,
        scoring_method=scoring_method,
        )
        

        data_for_training, data_for_prediction = get_ML_train_and_prediction_data(
            all_input_df=combined_df,
            name_cid_dict=name_cid_dict,
            out_dir=out_dir,
            input_columns=input_columns,
            scoring_method=scoring_method,
        )

        predicted_df = classify_data(
            data_for_prediction=data_for_prediction,
            data_for_training=data_for_training,
            model_names=ml_model_names,
            optimizer_name=optimizer_name,
            out_dir=out_dir,
            rand_labels=rand_labels,
            scoring_metrics=scoring_metrics,
            validation_cv=validation_cv,
        )
        return predicted_df

    else:
        raise ValueError(
            f"Invalid method: {method}. Please choose from: Embedding_only, Embedding_then_ML, Data_extraction_then_ML, physicochemical_data_and_embedding_then_ML"
        )


def get_ML_train_and_prediction_data(
    all_input_df: pd.DataFrame,
    name_cid_dict: dict,
    out_dir: str | Path,
    input_columns: list[str],
    scoring_method: str = "ZIP",
):
    """
    Split the data into training and prediction sets

    :param all_input_df: a DataFrame containing the feature values for all drugs.
    :param out_dir: the directory where the output files will be saved.
    :param input_columns: a list of the column names containing the feature names.
    :param scoring_method: the synergy scoring method. Defaults to "ZIP".

    :return: training and prediction sets as pandas DataFrames.
    """
    comb_ph = pd.read_csv(f"{out_dir}/combinations_in_kg.csv")
    comb_ph["label"] = comb_ph[scoring_method].apply(value_to_class)
    comb_ph["DrugPair_comb_ph"] = comb_ph[["Drug1_name", "Drug2_name"]].apply(
        lambda x: "-".join(sorted(x)), axis=1
    )
    all_input_df = all_input_df.rename(
        columns={
            "Drug1_name": "drug1_name",
            "Drug2_name": "drug2_name",
        }
    )
    all_input_df["DrugPair_all_drugs"] = all_input_df[
        ["drug1_name", "drug2_name"]
    ].apply(lambda x: "-".join(sorted(x)), axis=1)
    merged_df = pd.merge(
        all_input_df,
        comb_ph,
        left_on=["DrugPair_all_drugs"],
        right_on=["DrugPair_comb_ph"],
        how="left",
    )
    in_pharmacome = merged_df[(merged_df["DrugPair_comb_ph"].notnull())]
    not_in_pharmacome = merged_df[(merged_df["DrugPair_comb_ph"].isna())]

    final_columns_training = (
        [
            "Drug1_CID",
            "Drug2_CID",
            "Drug1_name",
            "Drug2_name",
        ]
        + input_columns
        + ["label"]
    )
    final_columns_prediction = (
        [
            "drug1_name",
            "drug2_name",
        ]
        + input_columns
    )

    train_df = in_pharmacome[final_columns_training]
    pred_df = not_in_pharmacome[final_columns_prediction]
    pred_df = pred_df.rename(
        columns={
            "drug1_name": "Drug1_name",
            "drug2_name": "Drug2_name",
        }
    )
    pred_df.loc[:,'Drug1_CID']= pred_df.loc[:,'Drug1_name'].apply(name_cid_dict.get)
    pred_df.loc[:,'Drug2_CID']= pred_df.loc[:,'Drug2_name'].apply(name_cid_dict.get)
    
    pred_df= pred_df[final_columns_training[:-1]]
    

    train_df.to_csv(f"{out_dir}/ML_training_dataset_with_names.csv", index=False)
    pred_df.to_csv(f"{out_dir}/ML_prediction_dataset_with_names.csv", index=False)

    first_column = input_columns[0]

    train_dataset = train_df.loc[:, first_column:"label"]
    train_dataset.to_csv(f"{out_dir}/ML_training_dataset.csv", index=False)

    pred_dataset = pred_df.loc[:, first_column:]
    pred_dataset.to_csv(f"{out_dir}/ML_prediction_dataset.csv", index=False)
    return train_dataset, pred_df


def value_to_class(value: float) -> int:
    """
    Convert synergy scores to classes.

    :param value: the synergy score.

    :return: the class label. -1 for negative scores, 0 for zero scores, and 1 for positive scores.
    """
    if value == 0:
        return 0
    elif value > 0:
        return 1
    else:
        return -1
