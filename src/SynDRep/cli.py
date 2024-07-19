# -*- coding: utf-8 -*-

"""Command line interface for SynDRep."""

import json
from typing import List
import click
import pandas as pd
from .combos_preparation import generate_enriched_kg
from .embedding import embed_and_predict
from .ML import classify_data
from .main_function import run_SynDRep


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
ml_models_option = click.option(
    "-m",
    "--model_names",
    multiple=True,
    type=str,
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

em_model_option = click.option(
    "-e",
    "--em_model_names",
    multiple=True,
    type=str,
    default=["TransE"],
    help='Models to use. can provide more than one of "TransE", "HolE", "TransR", "RotatE", "CompGCN", and "ComplEx".',
    required=False,
)

best_out_file_option = click.option(
    "--best_out_file",
    type=str,
    default="predictions_best.csv",
    help="Output file for best predictions.",
    required=False,
)

config_path_option = click.option(
    "--config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to the config file.",
    required=False,
)

filter_training_option = click.option(
    "-f",
    "--filter-training",
    type=bool,
    default=False,
    help="Filter training data from assesment of embedding models. Default: False",
    required=False,
)

get_embeddings_option = click.option(
    "--get-embeddings-data",
    type=bool,
    default=False,
    help="Get embeddings data. Default: False",
    required=False,
)

drug_class_name_option = click.option(
    "--drug-class-name",
    type=str,
    required=True,
    help="Name of the drug class to predict embeddings for.",
)

pred_reverse_option = click.option(
    "--pred_reverse",
    type=bool,
    default=True,
    help="Whether to add reverse predictions.",
    required=False,
)
subsplits_option = click.option(
    "--subsplits",
    type=bool,
    default=True,
    help="Whether to use subsplits for training and validation. Default: True",
    required=False,
)

kg_labels_file_option = click.option(
    "--kg-labels-file",
    type=click.Path(exists=True),
    help="Path to the KG labels file.",
    required=True,
)

name_cid_dict_option = click.option(
    "-n",
    "--name_cid_dict",
    type=click.Path(exists=True),
    help="Path to the name_cid_dict file.",
    required=False,
)

scoring_method_option = click.option(
    "-s",
    "--Scoring_method",
    type=click.Choice(["ZIP", "HSA", "Bliss", "Loewe"]),
    default="ZIP",
    help="Scoring method to use.",
    required=False,
)

combos_folder_option = click.option(
    "-c",
    "--combos_folder",
    type=click.Path(exists=True),
    help="Path to the combos folder.",
    required=True,
)

device_option = click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda",
    help="Device to use for computations",
    required=False,
)

method_option = click.option(
    "--method",
    type=click.Choice(["Embeeding_only", "Embeeding_then_ML", "Data_extraction_then_ML", "physicochemical_data_and_embedding_then_ML"]),
    default="Embeeding_only",
    help="Method to use for syndrep. Please choose from: Embeeding_only, Embeeding_then_ML, Data_extraction_then_ML, physicochemical_data_and_embedding_then_ML",
    required=False,
)

nBits_option = click.option(
    "--nBits",
    type=int,
    default=2048,
    help="Number of bits for the Morgan fingerprint.",
    required=False,
)

radius_option = click.option(
    "--radius",
    type=float,
    default=6,
    help="Radius for the Morgan fingerprint.",
    required=False,
)

@main.command()
@kg_file_option
@kg_drug_file_option
@out_dir_option
@name_cid_dict_option
@scoring_method_option
@combos_folder_option
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
@ml_models_option
@out_dir_option
@validation_cv_option
@scoring_metrics_option
@rand_labels_option
def classify(
    data_for_training,
    data_for_prediction,
    optimizer_name,
    ml_model_names,
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
        model_names=ml_model_names,
        out_dir=out_dir,
        validation_cv=validation_cv,
        scoring_metrics=scoring_metrics,
        rand_labels=rand_labels,
        *args,
    )
    return


@main.command
@kg_drug_file_option
@em_model_option
@kg_file_option
@out_dir_option
@best_out_file_option
@config_path_option
@filter_training_option
@get_embeddings_option
@drug_class_name_option
@pred_reverse_option
@subsplits_option
@kg_labels_file_option
def embedding_and_prediction(
    drug_class_name,
    em_models_names,
    kg_drug_file,
    kg_file,
    kg_labels_file,
    out_dir,
    best_out_file,
    config_path,
    filter_training,
    get_embeddings_data,
    pred_reverse,
    subsplits,
):
    """
    Embedding and prediction.
    """

    embed_and_predict(
        best_out_file=best_out_file,
        config_path=config_path,
        drug_class_name=drug_class_name,
        filter_training=filter_training,
        get_embeddings_data=get_embeddings_data,
        kg_drug_file=kg_drug_file,
        kg_file=kg_file,
        kg_labels_file=kg_labels_file,
        models_names=em_models_names,
        out_dir=out_dir,
        pred_reverse=pred_reverse,
        subsplits=subsplits,
    )


@main.command()
@best_out_file_option
@config_path_option
@device_option
@drug_class_name_option
@filter_training_option
@kg_drug_file_option
@kg_file_option
@kg_labels_file_option
@method_option
@ml_models_option
@nBits_option
@radius_option
@method_option
@nBits_option
@name_cid_dict_option
@optimizer_option
@out_dir_option
@pred_reverse_option
@radius_option
@rand_labels_option
@scoring_method_option
@scoring_metrics_option
@subsplits_option
@validation_cv_option
def run_syndrep(
    best_out_file,
    combos_folder,
    config_path,
    device,
    drug_class_name,
    em_models_names,
    filter_training,
    kg_drug_file,
    kg_file,
    kg_labels_file,
    method,
    ml_model_names,
    nBits,
    name_cid_dict,
    optimizer_name,
    out_dir,
    pred_reverse,
    radius,
    rand_labels,
    scoring_method,
    scoring_metrics,
    subsplits,
    validation_cv,
):

    run_SynDRep(
        best_out_file=best_out_file,
        combos_folder=combos_folder,
        config_path=config_path,
        device=device,
        drug_class_name=drug_class_name,
        em_models_names=em_models_names,
        filter_training=filter_training,
        kg_drug_file=kg_drug_file,
        kg_file=kg_file,
        kg_labels_file=kg_labels_file,
        method=method,
        ml_model_names=ml_model_names,
        nBits=nBits,
        name_cid_dict=name_cid_dict,
        optimizer_name=optimizer_name,
        out_dir=out_dir,
        pred_reverse=pred_reverse,
        radius=radius,
        rand_labels=rand_labels,
        scoring_method=scoring_method,
        scoring_metrics=scoring_metrics,
        subsplits=subsplits,
        validation_cv=validation_cv,
    )


if __name__ == "__main__":
    main()
