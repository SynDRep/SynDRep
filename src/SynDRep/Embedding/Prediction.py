# -*- coding: utf-8 -*-

"""wrapper for prediction functions"""

import pandas as pd

from tqdm import tqdm
from pykeen.predict import predict_target


def predict_diff_dataset(
    model,
    model_name,
    training_tf,
    main_test_df: pd.DataFrame,
    out_dir,
    kg_labels_file,
    best_out_file: str,
    predict_all=False,
    all_out_file: str = None,
    with_annotation=False,
    training_df=None,
    testing_df=None,
    validation_df=None,
    filter_training=False,
):

    labels = pd.read_table(kg_labels_file, dtype=str)
    labels_dict = dict(zip(labels["name"], labels["Type"]))
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
        if not all([training_df, testing_df, validation_df]):
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


def pred_manipulation(df, training_df, testing_df, validation_df):

    columns = ["source", "relation", "target"]

    training_df.columns = columns
    testing_df.columns = columns
    validation_df.columns = columns

    df_training = df_checker(df, training_df, "training")
    df_testing = df_checker(df_training, testing_df, "testing")
    df_final = df_checker(df_testing, validation_df, "validation")

    return df_final


def df_checker(query_df, reference_df, set_name):

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
