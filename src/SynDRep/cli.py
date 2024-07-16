# -*- coding: utf-8 -*-

"""Command line interface."""

import json
from typing import List
import click
import pandas as pd
from Combos_prparation.prepare_combos import generate_enriched_kg
from Embedding.Embedding import embed_and_predict
from ML.Classify import classify_data


@click.group()
def main() -> None:
    """Run SynDRep."""

kg_file_option = click.option(
    "-k",
    "--kg_file",
    type=click.Path(exists=True),
    help="Path to the KG tsv-file.",
    required=True,
)


kg_drug_file_option = click.option(
    "-d",
    "--kg_drug_file",
    type=click.Path(exists=True),
    help="Path to the KG drugs csv-file.",
    required=True,
)

out_dir_option = click.option(
    "-o",
    "--out_dir",
    type=click.Path(exists=False),
    help="Path to the output directory.",
    required=True,
)

data_for_training_option = click.option(
    "-t",
    "--data_for_training",
    type=click.Path(exists=True),
    help="Path to the data_for_trining csv-file.",
    required=True,
)
data_for_prediction_option = click.option(
    "-p",
    "--data_for_prediction",
    type=click.Path(exists=True),
    help="Path to the data_for_prediction csv-file.",
    required=True,
)
optimizer_option = click.option(
    "--optimizer_name",
    type=click.Choice(["grid_search", "random_search", "bayesian_search"]),
    default="grid_search",
    help="Optimizer to use.",
    required=False,
)
models_option = click.option(
    "-m",
    "--model_names",
    multiple=True,
    type= str,
    default=["random_forest"],
    help='Models to use. can provide more than one of "logistic_regression", "elastic_net", "svm", "random_forest", and "gradient_boost"',
    required=False,
)
validation_cv_option = click.option(
    "-v",
    "--validation_cv",
    type=int,
    default=5,
    help="Number of cross-validation folds to use.",
    required=False,
)
scoring_metrics_option = click.option(
    "-s",
    "--scoring_metrics",
    multiple=True,
    type=str,
    default=["roc_auc"],
    help='Scoring metric to use.can provide more than one of "accuracy", "f1_weighted", "f1", "roc_auc", "f1_macro", and "f1_micro"',
    required=False,
)
rand_labels_option = click.option(
    "-r",
    "--rand_labels",
    type=bool,
    default=False,
    help="Whether to schuffle the labels randomly.",
    required=False,
)

@main.command()
@kg_file_option
@kg_drug_file_option
@out_dir_option
@click.option(
    "-n",
    "--name_cid_dict",
    type=click.Path(exists=True),
    help="Path to the name_cid_dict file.",
    required=False,
)
@click.option(
    "-s",
    "--Scoring_method",
    type=click.Choice(["ZIP", "HSA", "Bliss", "Loewe"]),
    default="ZIP",
    help="Scoring method to use.",
    required=False,
)
@click.option(
    "-c",
    "--combos_folder",
    type=click.Path(exists=True),
    help="Path to the combos folder.",
    required=True,
)
def enriched_kg(
    kg_file,
    kg_drug_file,
    out_dir,
    combos_folder,
    name_cid_dict=None,
    scoring_method="ZIP",
):
    """Produces an enriched KG with drug-drug combinations."""
    if name_cid_dict:
        name_cid_dict = json.load(open(name_cid_dict))
    
    generate_enriched_kg(
        kg_file=kg_file,
        combos_folder=combos_folder,
        kg_drug_file=kg_drug_file,
        out_dir=out_dir,
        name_cid_dict=name_cid_dict,
        scoring_method=scoring_method,
    )


@main.command()
@data_for_training_option
@data_for_prediction_option
@optimizer_option
@models_option
@out_dir_option
@validation_cv_option
@scoring_metrics_option
@rand_labels_option
def classify(
    data_for_training,
    data_for_prediction,
    optimizer_name,
    model_names,
    out_dir,
    validation_cv,
    scoring_metrics,
    rand_labels,
    *args,
    
):
    """Classify data"""
    data_for_training = pd.read_csv(data_for_training)
    data_for_prediction = pd.read_csv(data_for_prediction)
    classify_data(
        data_for_training=data_for_training,
        data_for_prediction=data_for_prediction,
        optimizer_name=optimizer_name,
        model_names=model_names,
        out_dir=out_dir,
        validation_cv=validation_cv,
        scoring_metrics=scoring_metrics,
        rand_labels=rand_labels,
        *args,
    )
    return










if __name__ == "__main__":
    main()
