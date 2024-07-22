# -*- coding: utf-8 -*-
"""wrapper for prediction functions"""
import gzip
from pathlib import Path
from typing import Tuple

import pandas as pd
from pykeen.models.base import Model
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
from tqdm import tqdm


def predict_diff_dataset(
    best_out_file: str,
    kg_labels_file: str | Path,
    main_test_df: pd.DataFrame,
    model: Model,
    model_name: str,
    out_dir: str | Path,
    training_tf: TriplesFactory,
    all_out_file: str = None,
    filter_training: bool = False,
    predict_all: bool = False,
    subsplits: bool = True,
    testing_df: pd.DataFrame = None,
    training_df: pd.DataFrame = None,
    validation_df: pd.DataFrame = None,
    with_annotation: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make prediction using an embedding model.

    :param best_out_file: the path to the csv file containing best model predictions.
    :param kg_labels_file: the path to the csv file containing KG labels.
    :param main_test_df: the DataFrame containing the main test data.
    :param model: the embedding model to use.
    :param model_name: the name of the embedding model.
    :param out_dir: the output directory for the results.
    :param training_tf: the training triples factory.
    :param all_out_file: the path to save all predicted relations with score for a pair of drugs of the test set if predict_all is True. Defaults to None.
    :param filter_training: whether to filter the training data. Defaults to False.
    :param predict_all: whether to predict all relations with score for a pair of drugs. Defaults to False.
    :param subsplits: whether to use subsplits. Defaults to True.
    :param testing_df: the DataFrame containing testing data. Defaults to None.
    :param training_df: the DataFrame containing training data. Defaults to None.
    :param validation_df: the DataFrame containing validation data. Defaults to None.
    :param with_annotation: whether to add annotation to the prediction results. Defaults to False.

    :return: a DataFrame containing the prediction results for all entities and a DataFrame containing the best prediction results for each pair of drugs.
    """

    labels = pd.read_table(kg_labels_file, dtype=str)
    labels_dict = dict(zip(labels["Name"], labels["Type"]))
    columns = ["source", "relation", "target"]
    main_test_df.columns = columns

    df_list_all = []
    df_list_best = []
    for i, row in tqdm(main_test_df.iterrows(), total=len(main_test_df)):
        head = row["source"]
        tail = row["target"]
        pred = predict_target(
            model=model, head=str(head), tail=str(tail), triples_factory=training_tf
        ).df

        # add head and tail data
        pred["head_label"] = str(head)
        pred["tail_label"] = str(tail)
        pred["head_type"] = labels_dict.get(str(head))
        pred["tail_type"] = labels_dict.get(str(tail))

        # add to list of data frames
        df_list_all.append(pred)

        # add the best 1 to another df
        pred_best = pred.head(1)
        df_list_best.append(pred_best)

    df_all = pd.concat(df_list_all, ignore_index=True)
    df_best = pd.concat(df_list_best, ignore_index=True)
    if with_annotation:
        if any([training_df is None, testing_df is None, validation_df is None]):
            raise Exception(
                " not all paths for training, testing, and validation have been provided"
            )
        annot_dict = {}
        for x in ["df_all", "df_best"]:
            # annotation
            annot_dict[x] = pred_manipulation(
                df=eval(x),
                training_df=training_df,
                testing_df=testing_df,
                validation_df=validation_df,
                subsplits=subsplits,
                main_test_df=main_test_df,
            )
            if filter_training:
                annot_dict[x] = annot_dict[x][
                    annot_dict[x]["in_training"] != True
                ].drop(["in_training"], axis=1)

        annot_dict["df_best"].to_csv(
            f"{out_dir}/{model_name}/{best_out_file}", index=False
        )
        if predict_all:
            if all_out_file:
                annot_dict["df_all"].to_csv(
                    f"{out_dir}/{model_name}/{all_out_file}", index=False
                )
            else:
                raise Exception("all_out_file was not provided")
        print("files has been written... :)")
        return annot_dict["df_all"], annot_dict["df_best"]
    else:
        df_best.to_csv(f"{out_dir}/{model_name}/{best_out_file}", index=False)
        if predict_all:
            if all_out_file:
                df_all.to_csv(f"{out_dir}/{model_name}/{all_out_file}", index=False)
            else:
                raise Exception("all_out_file was not provided")
        print("files has been written... :)")
        return df_all, df_best


def pred_manipulation(
    df: pd.DataFrame,
    testing_df: pd.DataFrame,
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    main_test_df: pd.DataFrame = None,
    subsplits: bool = False,
) -> pd.DataFrame:
    """
    Adds columns to the df indicating whether the triples are in the testing_df, training_df, or validation_df

    :param df: DataFrame with triples to check.
    :param testing_df: DataFrame with testing triples.
    :param training_df: DataFrame with training triples.
    :param validation_df: DataFrame with validation triples.
    :param main_test_df: DataFrame with main_test triples (optional). Deaults to None.
    :param subsplits: boolean indicating whether subsplits were done do it adds main_test. Defaults to False.

    :return: DataFrame with an additional column indicating whether the triples are in the testing_df, training_df, or validation_df
    """

    columns = ["source", "relation", "target"]

    training_df.columns = columns
    testing_df.columns = columns
    validation_df.columns = columns

    df_training = df_checker(df, training_df, "training")
    df_testing = df_checker(df_training, testing_df, "testing")
    df_final = df_checker(df_testing, validation_df, "validation")

    if subsplits:
        df_final = df_checker(df_final, main_test_df, "main_test")

    return df_final


def df_checker(
    query_df: pd.DataFrame, reference_df: pd.DataFrame, set_name: str
) -> pd.DataFrame:
    """
    Adds a column to the query_df indicating whether the triples are in the reference_df

    :param query_df: DataFrame with triples to check
    :param reference_df: DataFrame with reference triples
    :param set_name: string indicating the set name (e.g., training, testing, validation)

    :return: DataFrame with an additional column indicating whether the triples are in the reference_df
    """
    query_df[["head_label", "relation_label", "tail_label"]] = query_df[
        ["head_label", "relation_label", "tail_label"]
    ].astype(str)

    reference_df[["source", "relation", "target"]] = reference_df[
        ["source", "relation", "target"]
    ].astype(str)

    query_df[f"in_{set_name}"] = False
    for i, head_label in enumerate(query_df["head_label"]):
        rel = query_df["relation_label"][i]
        tail_label = query_df["tail_label"][i]
        result_df = reference_df[
            (
                (reference_df["source"] == str(head_label))
                & (reference_df["relation"] == str(rel))
                & (reference_df["target"] == str(tail_label))
            )
        ]
        if len(result_df["source"].tolist()) > 0:
            query_df.loc[i, f"in_{set_name}"] = True
    return query_df


def gz_to_dict(path: str | Path) -> dict:
    """A function to get the dictionary out of gz file

    :param path: path to gz file

    :return: dictionary of gz file contents
    """
    with gzip.open(path, "rt", encoding="utf-8") as gz_file:
        df = pd.read_csv(gz_file, delimiter="\t")
    df["label"] = df["label"].astype(str)
    dct = dict(zip(df["label"], df["id"]))
    return dct
