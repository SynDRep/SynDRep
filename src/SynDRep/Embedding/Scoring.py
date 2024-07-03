# -*- coding: utf-8 -*-

"""Wrapper fro scoring functions"""


import numpy as np
import json
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def percents_true_predictions(df):

    
    in_testing_count = df["in_testing"].value_counts().get(True, 0)
    in_training_count = df["in_training"].value_counts().get(True, 0)
    in_validation_count = df["in_validation"].value_counts().get(True,0)
    in_main_test_count = df["in_main_test"].value_counts().get(True,0) if 'in_main_test' in df.columns else 0
    t_no = in_testing_count + in_training_count + in_validation_count + in_main_test_count
    
    percent = t_no * 100 / len(df)

    return percent


def multiclass_score_func(prediction_best_df, actual_test_df):
    """Calculate the multiclass roc score"""

    # get auc for all rel
    y_pred_rel = prediction_best_df["relation_label"].values
    y_rel = actual_test_df["relation"].values
    roc_rel = get_roc(y_pred_rel, y_rel)

    return roc_rel


def get_roc(y_pred, y):
    classes = np.unique(y)
    n_classes = len(classes)

    y = preprocessing.label_binarize(y, classes=classes)
    y_pred = preprocessing.label_binarize(y_pred, classes=classes)

    metric = 0

    for label in range(n_classes):
        metric += roc_auc_score(y[:, label], y_pred[:, label])

    metric = metric / n_classes

    return metric


def mean_hits(results_json):

    with open(results_json, "r") as f1:
        model_dict = json.load(f1)
    mean = model_dict["metrics"]["both"]["realistic"]["adjusted_arithmetic_mean_rank"]
    hits = model_dict["metrics"]["both"]["realistic"]["hits_at_10"]
    return mean, hits


def draw_graph(df, title, xlabel, ylabel, path):
    
    df.plot(kind="bar", figsize=(10, 6))
    plt.title(title, fontweight="bold")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    plt.savefig(path)
