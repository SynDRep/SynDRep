# -*- coding: utf-8 -*-

"""Wrapper fro scoring functions"""


from pathlib import Path
from typing import Tuple
import numpy as np
import json
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def percents_true_predictions(df: pd.DataFrame)-> float:
    """
    Calculate the percentage of true predictions for all relations.

    :param df: DataFrame with in_testing, in_training, in_validation.
    
    :return: Percentage of true predictions for all relations.
    """

    in_testing_count = df["in_testing"].value_counts().get(True, 0)
    in_training_count = df["in_training"].value_counts().get(True, 0)
    in_validation_count = df["in_validation"].value_counts().get(True, 0)
    in_main_test_count = (
        df["in_main_test"].value_counts().get(True, 0)
        if "in_main_test" in df.columns
        else 0
    )
    t_no = (
        in_testing_count + in_training_count + in_validation_count + in_main_test_count
    )

    percent = t_no * 100 / len(df)

    return percent


def multiclass_score_func(
    prediction_best_df:pd.DataFrame,
    actual_test_df:pd.DataFrame,
    )-> float:
    """
    Calculate the roc for all relations.

    :param prediction_best_df: DataFrame with predicted scores.
    :param actual_test_df: DataFrame with ground truth labels.

    :return: ROC score for all relations.
    """

    # get auc for all rel
    y_pred_rel = prediction_best_df["relation_label"].values
    y_rel = actual_test_df["relation"].values
    roc_rel = get_roc(y_pred_rel, y_rel)

    return roc_rel


def get_roc(y_pred: np.array, y: np.array) -> float:
    """
    Calculate the roc for binary classification.

    :param y_pred: Predicted scores.
    :param y: Ground truth labels.

    :return: ROC score.
    """
    classes = np.unique(y)
    n_classes = len(classes)

    y = preprocessing.label_binarize(y, classes=classes)
    y_pred = preprocessing.label_binarize(y_pred, classes=classes)

    metric = 0

    for label in range(n_classes):
        metric += roc_auc_score(y[:, label], y_pred[:, label])

    metric = metric / n_classes

    return metric


def mean_hits(results_json: str | Path) -> Tuple[float, float]:
    """
    Load results JSON and calculate the mean and hits at 10.

    :param results_json: Path to the results JSON file.

    :return: Mean and hits at 10 .

    """

    with open(results_json, "r") as f1:
        model_dict = json.load(f1)
    mean = model_dict["metrics"]["both"]["realistic"]["adjusted_arithmetic_mean_rank"]
    hits = model_dict["metrics"]["both"]["realistic"]["hits_at_10"]
    return mean, hits


def draw_graph(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    path: str | Path,
) -> None:
    """
    Draw a bar chart for the given DataFrame.

    :param df: DataFrame to draw.
    :param title: Title of the chart.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param path: Path to save the chart image.

    :return: None
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    plt.savefig(path)
    plt.close()
    return
