# -*- coding: utf-8 -*-

"""Wrapper fro scoring functions"""


import numpy as np
import json
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import numpy as np


def percents_true_prdictions(df):

    trues = []
    percents = {}

    t_no = df["in_testing"].value_counts().get(True, 0) + df[
        "in_validation"
    ].value_counts().get(True, 0)
    trues.append(t_no)
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
