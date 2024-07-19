# -*- coding: utf-8 -*-
"""main SynDRep function"""
import json
from pathlib import Path
from typing import List

import pandas as pd
from .combos_preparation import generate_enriched_kg
from .embedding import compare_embeddings, embed_and_predict
from .ML import classify_data
from .drug_data import get_graph_and_physicochem_properties
from .drug_data import get_physicochem_prop
from tqdm import tqdm


def run_SynDRep(
    em_models_names: List[str],
    kg_file,
    out_dir,
    combos_folder: str,
    kg_drug_file: str,
    optimizer_name: str,
    ml_model_names: List[str],
    validation_cv: int,
    scoring_metrics: List[str],
    rand_labels: bool,
    kg_labels_file,
    drug_class_name,
    all_drug_drug_predictions=False,
    all_drug_prop_dict: dict = None,
    all_out_file: str = None,
    best_out_file: str = "predictions_best.csv",
    config_path=None,
    device="cuda",
    filter_training=False,
    method: str = "Embeeding_only",
    nBits: int = 2048,
    name_cid_dict: dict = None,
    pred_reverse=False,
    predict_all=False,
    radius: int = 6,
    scoring_method: str = "ZIP",
    sorted_predictions=True,
    subsplits=True,
):
    generate_enriched_kg(
        combos_folder := combos_folder,
        kg_drug_file=kg_drug_file,
        kg_file=kg_file,
        name_cid_dict=name_cid_dict,
        out_dir=out_dir,
        scoring_method=scoring_method,
    )
    if method == "Embeeding_only":

        prediction_all, prediction_best = embed_and_predict(
            kg_drug_file=kg_drug_file,
            models_names=em_models_names,
            kg_file=f"{out_dir}/enriched_kg.tsv",
            out_dir=out_dir,
            best_out_file=best_out_file,
            config_path=config_path,
            subsplits=subsplits,
            kg_labels_file=kg_labels_file,
            drug_class_name=drug_class_name,
            predict_all_test=predict_all,
            all_out_file=all_out_file,
            filter_training=filter_training,
            get_embeddings_data=False,
            pred_reverse=pred_reverse,
            sorted_predictions=sorted_predictions,
            all_drug_drug_predictions=all_drug_drug_predictions,
        )
        return prediction_all, prediction_best

    elif method == "Embeeding_then_ML":
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
            filter_training=filter_training,
            drug_class_name=drug_class_name,
            get_embeddings_data=True,
            kg_file=kg_file,
            kg_labels_file=kg_labels_file,
            models_names=em_models_names,
            out_dir=out_dir,
            subsplits=subsplits,
            predict_all=predict_all,
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
        combinations = list(combinations(embeddings.iterrows(), 2))
        # Iterate through all combinations of node pairs
        for (idx1, row1), (idx2, row2) in tqdm(combinations):
            combined_row = {}

            # Add the node ids to the combined row
            combined_row["Drug1_name"] = row1["entity"]
            combined_row["Drug2_name"] = row2["entity"]
            # Add the node cids to the combined row
            combined_row["Drug1_CID"] = combined_row["Drug1_name"].apply(
                name_cid_dict.get
            )
            combined_row["Drug2_CID"] = combined_row["Drug2_name"].apply(
                name_cid_dict.get
            )
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
        data_for_training, data_for_prediction = get_ML_train_and_preidction_data(
            all_input_df=combined_df,
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

        combined_df.to_csv(
            f"{best_model}_drug_combinations_physicochemical_and_graph_data.csv",
            index=False,
        )

        input_columns = [
            col
            for col in combined_df.columns
            if ("name" not in col) and ("CID" not in col)
        ]

        data_for_training, data_for_prediction = get_ML_train_and_preidction_data(
            all_input_df=combined_df,
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
            filter_training=filter_training,
            drug_class_name=drug_class_name,
            get_embeddings_data=True,
            kg_file=kg_file,
            kg_labels_file=kg_labels_file,
            models_names=em_models_names,
            out_dir=out_dir,
            subsplits=subsplits,
            predict_all=predict_all,
        )
        embeddings = pd.read_csv(
            f"{out_dir}/{best_model}/{best_model}_embeddings_drugs.csv"
        )
        # Get the list of embedding columns
        embedding_columns = [
            col for col in embeddings.columns if col.startswith("embedding_")
        ]

        # Create an empty DataFrame for the combined embeddings
        embedding_rows = []
        ph_ch_rows = []
        combinations = list(combinations(embeddings.iterrows(), 2))
        # Iterate through all combinations of node pairs
        for (idx1, row1), (idx2, row2) in tqdm(combinations):
            embedding_row = {}

            # Add the embeddings to the combined row
            for col in embedding_columns:
                embedding_row[f"Drug1_{col}"] = row1[col]
                embedding_row[f"Drug2_{col}"] = row2[col]
            embedding_rows.append(embedding_row)

            # get physicochemical data
            ph_ch_row = get_physicochem_prop(
                drug1_name=row1["entity"],
                drug2_name=row2["entity"],
                out_dir=out_dir,
                all_drug_prop_dict=all_drug_prop_dict,
                device=device,
                radius=radius,
            )
            ph_ch_rows.append(ph_ch_row)

        embedding_df = pd.DataFrame(embedding_rows)
        ph_ch_df = pd.concat(ph_ch_rows)

        combined_df = pd.concat([ph_ch_df, embedding_df], axis=1).reset_index(
            inplace=True, drop=True
        )

        combined_df.to_csv(
            f"{best_model}_drug_combinations_physicochemical_and_embeddings.csv",
            index=False,
        )

        input_columns = [
            col
            for col in combined_df.columns
            if ("name" not in col) and ("CID" not in col)
        ]

        data_for_training, data_for_prediction = get_ML_train_and_preidction_data(
            all_input_df=combined_df,
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
            f"Invalid method: {method}. Please choose from: Embeeding_only, Embeeding_then_ML, Data_extraction_then_ML, physicochemical_data_and_embedding_then_ML"
        )


def get_ML_train_and_preidction_data(
    all_input_df: pd.DataFrame,
    out_dir: str | Path,
    input_columns: list[str],
    scoring_method: str = "ZIP",
):
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

    final_columns = (
        [
            "Drug1_CID",
            "Drug2_CID",
            "Drug1_name",
            "Drug2_name",
        ]
        + input_columns
        + ["label"]
    )

    train_df = in_pharmacome[final_columns]
    pred_df = not_in_pharmacome[final_columns]

    train_df.to_csv(f"{out_dir}/ML_training_dataset_with_names.csv", index=False)
    pred_df.to_csv(f"{out_dir}/ML_prediction_dataset_with_names.csv", index=False)

    first_column = input_columns[0]

    train_dataset = train_df.loc[:, first_column:"label"]
    train_dataset.to_csv(f"{out_dir}/ML_training_dataset.csv", index=False)

    pred_dataset = pred_df.loc[:, first_column:]
    pred_dataset.to_csv(f"{out_dir}/ML_prediction_dataset.csv", index=False)
    return train_dataset, pred_dataset


def value_to_class(value):
    if value == 0:
        return 0
    elif value > 0:
        return 1
    else:
        return -1
