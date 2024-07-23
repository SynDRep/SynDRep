# -*- coding: utf-8 -*-

"""Wrapper for scoring functions."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from numpy import mean


def multiclass_score_func(y, y_pred, metric_func, **kwargs):
    """Calculate the multiclass metric score of any sklearn metric.
    
    :param y: Ground truth labels
    :param y_pred: Predicted labels
    :param metric_func: sklearn metric function
    :param kwargs: Additional keyword arguments for the metric function
    
    :return: Metric score
    """
    classes = np.unique(y)
    n_classes = len(classes)

    if n_classes == 2:
        return metric_func(y, y_pred)

    y = preprocessing.label_binarize(y, classes=classes)
    y_pred = preprocessing.label_binarize(y_pred, classes=classes)

    metric = 0

    for label in range(n_classes):
        metric += metric_func(y[:, label], y_pred[:, label], **kwargs)

    metric /= n_classes

    return metric


def draw_graph(model_names: str, out_dir: str|Path, metric: str="roc_auc"):
    """Draw a boxplot for different ML models.
    
    :param model_names: List of ML model names
    :param out_dir: Output directory for the output file
    :param metric: Metric to be plotted. Defaults to "roc_auc"
    
    :return: A dictionary containing mean scores for each model
    """

    data_dict = {}
    models_mean = {}

    for model_name in model_names:
        cv_results = json.load(
            open(f"{out_dir}/{model_name}/cross_validation_results.json")
        )

        data_dict[model_name] = cv_results[f"test_{metric}"]
        models_mean[model_name] = round(mean(cv_results[f"test_{metric}"]), 4)

    # Convert the dictionary values to a list of lists for boxplot
    data_list = list(data_dict.values())

    # Create a boxplot with spaces between boxes
    box_width = 0.8  # Adjust the width of the boxes as needed
    box_positions = [
        2 * i + 1 for i in range(len(data_dict.keys()))
    ]  # Set positions every two units
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data_list, labels=data_dict.keys(), positions=box_positions, widths=box_width
    )

    plt.title("Boxplot for different ML Models", fontweight="bold")
    plt.xlabel("Models", fontweight="bold")
    plt.ylabel(f"{metric}".upper(), fontweight="bold")
    plt.savefig(f"{out_dir}/{metric}.png")
    return models_mean
