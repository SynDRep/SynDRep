# -*- coding: utf-8 -*-
"""do the embedding of enriched KG"""
import json
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from pykeen.hpo.hpo import HpoPipelineResult, hpo_pipeline_from_config
from pykeen.models.base import Model
from pykeen.pipeline import PipelineResult, pipeline_from_config
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
from tqdm import tqdm

from SynDRep.embedding.models_config import get_config
from SynDRep.embedding.prediction import gz_to_dict, predict_diff_dataset
from SynDRep.embedding.scoring import (
    draw_graph,
    mean_hits,
    multiclass_score_func,
    percents_true_predictions,
)


def embed_and_predict(
    drug_class_name: str,
    kg_drug_file: str | Path,
    kg_file: str | Path,
    kg_labels_file: str | Path,
    models_names: List[str],
    out_dir: str | Path,
    all_drug_drug_predictions: bool = False,
    all_out_file: str = None,
    best_out_file: str = "predictions_best.csv",
    config_path: str | Path = None,
    filter_training: bool = False,
    get_embeddings_data: bool = False,
    pred_reverse: bool = True,
    predict_all_test: bool = False,
    sorted_predictions: bool = True,
    subsplits: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make embedding using different models and determines the best nodel then use it for prediction.

    :param drug_class_name: the string indicating the drug class in th KG.
    :param kg_drug_file: the path to the csv file containing drug names.
    :param kg_file: the path to the original KG file.
    :param kg_labels_file: the path to the csv file containing labels of the nodes in the graph.
    :param models_names: a list of model names.
    :param out_dir: the path to the output directory.
    :param all_drug_drug_predictions: a boolean indicating whether to produce the all predicted relations with score for a pair of drugs or just the best scoring relation. Defaults to False.
    :param all_out_file: the path to save all predicted relations with score for a pair of drugs of the test set if predict_all_test is True. Defaults to None.
    :param best_out_file: the path to save the best predicted relation with score for a pair of drugs of the test set. Defaults to "predictions_best.csv".
    :param config_path: the path to the configuration file for hyperparameter optimization. Defaults to None.
    :param filter_training: a boolean indicating whether to filter the training data. Defaults to False.
    :param get_embeddings_data: a boolean indicating whether to get embeddings data. Defaults to False.
    :param pred_reverse: a boolean indicating whether to predict  the reverse relations. Defaults to True.
    :param predict_all_test: a boolean indicating whether to predict all test set relations or just the best scoring relation. Defaults to False.
    :param sorted_predictions: a boolean indicating whether to sort the predictions. Defaults to True.
    :param subsplits: a boolean indicating whether to use subsplits for training. Defaults to True.

    :return: DataFrame containing all the relation prediction for each pair of drugs, and a DataFrame containing the best predicted relation for each pair drugs.
    """
    best_model = compare_embeddings(
        all_out_file=all_out_file,
        best_out_file=best_out_file,
        config_path=config_path,
        drug_class_name=drug_class_name,
        enriched_kg=True,
        filter_training=filter_training,
        get_embeddings_data=get_embeddings_data,
        kg_file=kg_file,
        kg_labels_file=kg_labels_file,
        models_names=models_names,
        out_dir=out_dir,
        predict_all=predict_all_test,
        subsplits=subsplits,
    )

    model = torch.load(
        f"{out_dir}/{best_model}/{best_model}_best_model_results/trained_model.pkl"
    )

    entity_to_id = gz_to_dict(
        f"{out_dir}/{best_model}/{best_model}_best_model_results/training_triples/entity_to_id.tsv.gz"
    )
    relation_to_id = gz_to_dict(
        f"{out_dir}/{best_model}/{best_model}_best_model_results/training_triples/relation_to_id.tsv.gz"
    )

    if subsplits:
        tf = TriplesFactory.from_path(
            f"{out_dir}/train_data_ss.tsv",
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )
    else:
        tf = TriplesFactory.from_path(
            f"{out_dir}/train_data.tsv",
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

    drugs = pd.read_csv(kg_drug_file)["Drug_name"].tolist()
    combs = list(combinations(drugs, 2))
    df_list_all = []
    df_list_best = []

    for comb in tqdm(combs):
        head, tail = comb
        pred_df = predict_target(
            model=model, head=str(head), tail=str(tail), triples_factory=tf
        ).df
        pred_df.loc[:, "Drug1_name"] = str(head)
        pred_df.loc[:, "Drug2_name"] = str(tail)
        # add to list of data frames
        df_list_all.append(pred_df)
        # add the best 1 to another df
        pred_df_best = pred_df.head(1)
        df_list_best.append(pred_df_best)

        if pred_reverse:
            reverse_pred_df = predict_target(
                model=model, head=str(tail), tail=str(head), triples_factory=tf
            ).df
            reverse_pred_df.loc[:, "Drug1_name"] = str(tail)
            reverse_pred_df.loc[:, "Drug2_name"] = str(head)
            # add to list of data frames
            df_list_all.append(reverse_pred_df)
            # add the best 1 to another df
            reverse_pred_df_best = reverse_pred_df.head(1)
            df_list_best.append(reverse_pred_df_best)

    df_all = pd.concat(df_list_all, ignore_index=True).reset_index(drop=True)
    df_best = pd.concat(df_list_best, ignore_index=True).reset_index(drop=True)

    if sorted_predictions:
        df_best.sort_values(by=["score"], ascending=False, inplace=True)

    if all_drug_drug_predictions:
        df_all.to_csv(
            f"{out_dir}/{best_model}/{best_model}_drug_drug_predictions_all.csv",
            index=False,
        )
    df_best.to_csv(
        f"{out_dir}/{best_model}/{best_model}_drug_drug_predictions_best.csv",
        index=False,
    )
    return df_all, df_best


def compare_embeddings(
    drug_class_name: str,
    kg_file: str | Path,
    kg_labels_file: str | Path,
    models_names: List[str],
    out_dir: str | Path,
    all_out_file: str = None,
    best_out_file: str = "predictions_best.csv",
    config_path: str | Path = None,
    enriched_kg: bool = False,
    filter_training: bool = False,
    get_embeddings_data: bool = False,
    predict_all: bool = False,
    subsplits: bool = True,
) -> str:
    """
    Compare embeddings using different models and determine the best one.

    :param drug_class_name: the string indicating the drug class in th KG.
    :param kg_file: the path to the KG file.
    :param kg_labels_file: the path to the KG labels file.
    :param models_names: a list of model names to compare.
    :param out_dir: the output directory.
    :param all_out_file: the file to save all predictions.Defaults to None.
    :param best_out_file: the file to save the best predictions. Defaults to "predictions_best.csv".
    :param config_path: the path to the configuration file. Defaults to None.
    :param enriched_kg: whether to use enriched KG. Defaults to False.
    :param filter_training: whether to filter the training data. Defaults to False.
    :param get_embeddings_data: whether to get embeddings data. Defaults to False.
    :param predict_all: whether to produce the all predicted relations with score for a pair of drugs or just the best scoring relation. Defaults to False.
    :param subsplits: whether to use subsplits. Defaults to True.

    :return: the name of the best model.
    """

    results = []

    for model_name in models_names:
        embedding_results = kg_embedding(
            all_out_file=all_out_file,
            best_out_file=best_out_file,
            config_path=config_path,
            drug_class_name=drug_class_name,
            filter_training=filter_training,
            kg_file=kg_file,
            kg_labels_file=kg_labels_file,
            model_name=model_name,
            out_dir=out_dir,
            predict_all=predict_all,
            subsplits=subsplits,
        )
        result = [model_name]
        result.extend(embedding_results[1:])
        results.append(result)

    columns = [
        "model",
        "Percentage of true predictions for all relations",
        "adjusted_arithmetic_mean_rank",
        "hits_at_10",
        "roc_auc for all relations",
        f"Percentage of true predictions for {drug_class_name}-{drug_class_name} relations",
    ]
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(f"{out_dir}/Models_results.csv", index=False)
    if enriched_kg:
        max_score_row = results_df.loc[
            results_df[
                f"Percentage of true predictions for {drug_class_name}-{drug_class_name} relations"
            ].idxmax()
        ]
    else:
        max_score_row = results_df.loc[
            results_df["Percentage of true predictions for all relations"].idxmax()
        ]
    best_model = max_score_row["model"]
    df = pd.read_csv(f"{out_dir}/Models_results.csv")
    df.set_index("model", inplace=True)
    for metric in columns[1:]:
        draw_graph(
            df=df[metric],
            title=metric.replace("_", " ") + " for all models",
            xlabel="Models",
            ylabel=metric.replace("_", " "),
            path=f"{out_dir}/{metric}.jpg",
        )

    if get_embeddings_data:
        get_embeddings(
            model=torch.load(
                f"{out_dir}/{best_model}/{best_model}_best_model_results/trained_model.pkl"
            ),
            training_tf=embedding_results[0],
            kg_labels_file=kg_labels_file,
            output_all=f"{out_dir}/{best_model}/{best_model}_embeddings_all.csv",
            output_drugs=f"{out_dir}/{best_model}/{best_model}_embeddings_drugs.csv",
        )

    return best_model


def kg_embedding(
    drug_class_name: str,
    kg_file: str | Path,
    kg_labels_file: str | Path,
    model_name: str,
    out_dir: str | Path,
    all_out_file: str = None,
    best_out_file: str = "predictions_best.csv",
    config_path: str | Path = None,
    filter_training: bool = False,
    predict_all: bool = False,
    subsplits: bool = True,
) -> Tuple[TriplesFactory, float, float, float, float, float]:
    """
    Embeds the knowledge graph using the specified model and performs KG evaluation.

    :param drug_class_name: the string indicating the drug class in th KG.
    :param kg_file: the path to the KG file.
    :param kg_labels_file: the path to the KG labels file.
    :param model_name: the name of the model to use for embedding.
    :param out_dir: the output directory for the results.
    :param all_out_file: the output file for all predictions. Defaults to None.
    :param best_out_file: the output file for the best predictions. Defaults to "predictions_best.csv".
    :param config_path: the path to the configuration file for the model. Defaults to None.
    :param filter_training: whether to filter the training data. Defaults to False.
    :param predict_all: whether to produce the all predicted relations with score for a pair of drugs or just the best scoring relation. Defaults to False.
    :param subsplits: whether to use subsplits. Defaults to True.

    :return: a TriplesFactory object for the training data, the percentage of true predictions for all relations, the adjusted arithmetic mean rank, hits@10, ROC AUC for all relations, and the percentage of true predictions for the specified drug-drug relation
    """

    # make the splits
    if subsplits:
        (
            train_df,
            test_df,
            validation_df,
            train_tf,
            test_tf,
            validation_tf,
            main_test_df,
        ) = create_data_splits(
            kg_file=kg_file,
            out_dir=out_dir,
            kg_labels_file=kg_labels_file,
            drug_class_name=drug_class_name,
            subsplits=True,
        )
    else:
        (
            train_df,
            test_df,
            validation_df,
            train_tf,
            test_tf,
            validation_tf,
            main_test_df,
        ) = create_data_splits(
            kg_file=kg_file,
            out_dir=out_dir,
            kg_labels_file=kg_labels_file,
            drug_class_name=drug_class_name,
            subsplits=False,
        )

        main_test_df = pd.concat([test_df, validation_df], ignore_index=True)

    if config_path:
        config = json.load(open(config_path, "r"))
    else:
        config = get_config(model_name=model_name)

    # run hpo
    run_hpo(
        config=config,
        train_tf=train_tf,
        test_tf=test_tf,
        validation_tf=validation_tf,
        model_name=model_name,
        out_dir=out_dir,
    )

    # run pipeline
    pipeline_results = run_pipeline(
        train_tf, test_tf, validation_tf, model_name, out_dir
    )

    # prediction

    predict_diff_dataset(
        model=pipeline_results.model,
        model_name=model_name,
        training_tf=train_tf,
        main_test_df=main_test_df,
        out_dir=out_dir,
        kg_labels_file=kg_labels_file,
        best_out_file=best_out_file,
        with_annotation=True,
        subsplits=subsplits,
        training_df=train_df,
        testing_df=test_df,
        validation_df=validation_df,
        filter_training=filter_training,
        predict_all=predict_all,
        all_out_file=all_out_file,
    )
    # scoring
    best_test_pred = pd.read_csv(f"{out_dir}/{model_name}/{best_out_file}")
    percent_true = percents_true_predictions(best_test_pred)
    mean, hits = mean_hits(
        f"{out_dir}/{model_name}/{model_name}_best_model_results/results.json"
    )
    roc = multiclass_score_func(best_test_pred, main_test_df)
    results_dict = {
        "Percentage of true predictions for all relations": percent_true,
        "roc_auc for all relations": roc,
        "adjusted_arithmetic_mean_rank": mean,
        "hits_at_10": hits,
    }
    json.dump(
        results_dict,
        open(
            f"{out_dir}/{model_name}/{model_name}_best_model_results/{model_name}_scoring_results.json",
            "w",
        ),
        indent=4,
    )
    # specif types testing set predictions

    sp_test_df = pd.read_table(
        f"{out_dir}/test_{drug_class_name}_{drug_class_name}.tsv",
        header=None,
        names=["source", "relation", "target"],
    )
    predict_diff_dataset(
        model=pipeline_results.model,
        model_name=model_name,
        training_tf=train_tf,
        main_test_df=sp_test_df,
        out_dir=out_dir,
        kg_labels_file=kg_labels_file,
        best_out_file=f"{drug_class_name}_{drug_class_name}_{best_out_file}",
        with_annotation=True,
        subsplits=subsplits,
        training_df=train_df,
        testing_df=test_df,
        validation_df=validation_df,
        filter_training=filter_training,
        predict_all=predict_all,
        all_out_file=f"{drug_class_name}_{drug_class_name}_{all_out_file}",
    )

    best_test_pred_sp_type = pd.read_csv(
        f"{out_dir}/{model_name}/{drug_class_name}_{drug_class_name}_{best_out_file}"
    )
    percent_true_sp_type = percents_true_predictions(best_test_pred_sp_type)

    results_dict = {
        "Percentage of true predictions for all relations": percent_true,
        "roc_auc for all relations": roc,
        "adjusted_arithmetic_mean_rank": mean,
        "hits_at_10": hits,
        f"Percentage of true predictions for {drug_class_name}-{drug_class_name} relations": percent_true_sp_type,
    }
    json.dump(
        results_dict,
        open(
            f"{out_dir}/{model_name}/{model_name}_best_model_results/{model_name}_scoring_results.json",
            "w",
        ),
        indent=4,
    )

    return (
        train_tf,
        percent_true,
        mean,
        hits,
        roc,
        percent_true_sp_type,
    )


def get_embeddings(
    kg_labels_file: str | Path,
    model: Model,
    output_all: str,
    output_drugs: str,
    training_tf: TriplesFactory,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get embeddings for all entities in the knowledge graph.

    :param kg_labels_file: the path to the KG labels file.
    :param model: the trained model.
    :param output_all: the path to save the embeddings for all entities.
    :param output_drugs: the path to save the embeddings for drug entities only.
    :param training_tf: the training triples factory.

    :return: DataFrame containing all entity embeddings and drug embeddings.
    """

    # get embeddings

    entity_embeddings_array = (
        model.entity_representations[0](indices=None).detach().cpu().numpy().real
    )
    entities = training_tf.entity_id_to_label
    entity_ids = range(model.num_entities)
    embedding_columns = [
        f"embedding_{i}" for i in range(entity_embeddings_array.shape[1])
    ]
    data = {
        col: entity_embeddings_array[:, i] for i, col in enumerate(embedding_columns)
    }
    data["entity"] = [entities[entity_id] for entity_id in entity_ids]
    embeddings_df = pd.DataFrame(data)

    # adding labels
    labels = pd.read_table(kg_labels_file, dtype=str)
    name_type_dict = dict(zip(labels["Name"], labels["Type"]))

    embeddings_df["entity"] = embeddings_df["entity"].astype(str)
    embeddings_df["entity_type"] = embeddings_df["entity"].apply(name_type_dict.get)
    embeddings_df = embeddings_df[["entity", "entity_type"] + embedding_columns]
    embeddings_df.to_csv(output_all, index=False)

    drug_embeddings = embeddings_df[embeddings_df["entity_type"] == "Drug"]
    drug_embeddings.to_csv(output_drugs, index=False)
    return embeddings_df, drug_embeddings


def run_pipeline(
    model_name: str,
    out_dir: str | Path,
    train_tf: TriplesFactory,
    test_tf: TriplesFactory,
    validation_tf: TriplesFactory,
) -> PipelineResult:
    """
    Run the pipeline on the given triples factories.

    :param model_name: the name of the model
    :param out_dir: the output directory for the pipeline results
    :param train_tf: the training triples factory object
    :param test_tf: the testing triples factory object
    :param validation_tf: the validation triples factory object

    :return: the PipelineResult object containing the pipeline results
    """
    # run pipeline
    config = json.load(
        open(
            f"{out_dir}/{model_name}/{model_name}_hpo_results/best_pipeline/pipeline_config.json",
            "r",
        )
    )
    for data in ["training", "testing", "validation"]:
        config["pipeline"].pop(data, None)
    pipeline_result = pipeline_from_config(
        config=config, training=train_tf, validation=validation_tf, testing=test_tf
    )
    pipeline_result.save_to_directory(
        f"{out_dir}/{model_name}/{model_name}_best_model_results"
    )
    return pipeline_result


def run_hpo(
    config: dict,
    model_name: str,
    out_dir: str | Path,
    train_tf: TriplesFactory,
    test_tf: TriplesFactory,
    validation_tf: TriplesFactory,
) -> HpoPipelineResult:
    """
    Run hyperparameter optimization (HPO) for the given pipeline configuration.

    :param config: the pipeline configuration as a dictionary
    :param model_name: the name of the model
    :param out_dir: the output directory for the HPO results
    :param train_tf: the training TriplesFactory object
    :param test_tf: the testing TriplesFactory object
    :param validation_tf: the validation TriplesFactory object

    :return: the HpoPipelineResult object containing the HPO results
    """
    hpo_results = hpo_pipeline_from_config(
        config=config, training=train_tf, validation=validation_tf, testing=test_tf
    )
    Path(f"{out_dir}/{model_name}").mkdir(parents=True, exist_ok=True)
    hpo_results.save_to_directory(f"{out_dir}/{model_name}/{model_name}_hpo_results")
    return hpo_results


def create_data_splits(
    drug_class_name: str,
    kg_file: str | Path,
    kg_labels_file: str | Path,
    out_dir: str | Path,
    subsplits: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    TriplesFactory,
    TriplesFactory,
    TriplesFactory,
    pd.DataFrame,
]:
    """
    Create data splits for training, testing, and validation sets.

    :param drug_class_name: the string indicating the drug class in th KG
    :param kg_file: the path to the knowledge graph file
    :param kg_labels_file: the path to the KG labels file
    :param out_dir: the output directory to save the data splits
    :param subsplits: a boolean indicating whether to create subsplits

    :return: the training, testing, validation dataframes, and the TriplesFactory objects for each split
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Create a TriplesFactory object from your knowledge graph file
    kg_triples_factory = TriplesFactory.from_path(kg_file)
    train_df, test_df, validation_df, train_tf, test_tf, validation_tf = make_splits(
        kg_triples_factory=kg_triples_factory
    )

    # Split the knowledge graph into train, validation, and test sets
    while not nodes_relations_check(train_df, test_df, validation_df):
        train_df, test_df, validation_df, train_tf, test_tf, validation_tf = (
            make_splits(kg_triples_factory=kg_triples_factory)
        )

    print(
        "All nodes and relations in test and validation sets are there in the training set :)"
    )
    main_test_df = pd.concat([test_df, validation_df], ignore_index=True)
    if subsplits:

        (
            train_df_ss,
            test_df_ss,
            validation_df_ss,
            train_tf_ss,
            test_tf_ss,
            validation_tf_ss,
        ) = make_splits(kg_triples_factory=train_tf)
        while not nodes_relations_check(train_df_ss, test_df_ss, validation_df_ss):
            (
                train_df_ss,
                test_df_ss,
                validation_df_ss,
                train_tf_ss,
                test_tf_ss,
                validation_tf_ss,
            ) = make_splits(kg_triples_factory=train_tf)

        print(
            "All nodes and relations in test and validation subsets are there in the training set :)"
        )
        for data in [
            "train_data_ss",
            "test_data_ss",
            "validation_data_ss",
            "main_test_data",
        ]:
            triplets_to_file(out_folder=out_dir, df=eval(data.replace("data", "df")), data_type=data)

        print("all done :)")
        generate_drug_test_set(kg_labels_file=kg_labels_file, test_df=test_df, drug_class_name=drug_class_name, out_dir=out_dir)

        return (
            train_df_ss,
            test_df_ss,
            validation_df_ss,
            train_tf_ss,
            test_tf_ss,
            validation_tf_ss,
            main_test_df,
        )

    generate_drug_test_set(kg_labels_file=kg_labels_file, test_df=test_df, drug_class_name=drug_class_name, out_dir=out_dir)
    for data in ["train_data", "test_data", "validation_data"]:
        triplets_to_file(out_folder=out_dir, df=eval(data.replace("data", "df")), data_type=data)
    print("all done :)")
    return (
        train_df,
        test_df,
        validation_df,
        train_tf,
        test_tf,
        validation_tf,
        main_test_df,
    )


def make_splits(
    kg_triples_factory: TriplesFactory,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    TriplesFactory,
    TriplesFactory,
    TriplesFactory,
]:
    """
    Split the knowledge graph into train, validation, and test sets.

    :param kg_triples_factory: A TriplesFactory object containing the knowledge graph triples

    :return: DataFrames of training, test, and validation triples, and their TriplesFactory objects
    """

    train_tf, test_tf, validation_tf = kg_triples_factory.split(
        [0.8, 0.1, 0.1], random_state=42
    )
    train_df = triples_to_df(train_tf)
    test_df = triples_to_df(test_tf)
    validation_df = triples_to_df(validation_tf)

    return train_df, test_df, validation_df, train_tf, test_tf, validation_tf


# check all nodes and relations in test and validation are in training
def nodes_relations_check(
    train_df: pd.DataFrame, test_df: pd.DataFrame, validation_df: pd.DataFrame
) -> bool:
    """
    Check if all nodes and relations in test and validation are in the training set.

    :param train_df: DataFrame of training triples
    :param test_df: DataFrame of test triples
    :param validation_df: DataFrame of validation triples

    :return: True if all nodes and relations are in the training set, False otherwise
    """

    train_nodes, train_relations = get_nodes_and_relations(train_df)
    test_nodes, test_relations = get_nodes_and_relations(test_df)
    validation_nodes, validation_relations = get_nodes_and_relations(validation_df)

    check_list = []
    for nodes in [test_nodes, validation_nodes]:
        check_list.append(nodes.issubset(train_nodes))

    for relations in [test_relations, validation_relations]:
        check_list.append(relations.issubset(train_relations))
    print(
        "test_nodes, validation_nodes, test_relations, and validation_relations check, respectively: ",
        check_list,
    )
    if all(check_list):
        return True
    else:
        return False


def get_nodes_and_relations(df: pd.DataFrame) -> Tuple[set, set]:
    """Get all nodes and relations

    :param df: DataFrame of triples

    :return: a tuple of sets of nodes and relations
    """

    return set(df["source"]).union(df["target"]), set(df["relation"])


def triples_to_df(triple_tf: TriplesFactory) -> pd.DataFrame:
    """
    Map the IDs to labels and convert the triples to a DataFrame.

    :param triple_tf: the TriplesFactory object

    :return: a DataFrame with columns "source", "relation", and "target"
    """
    entity_id_to_label = triple_tf.entity_id_to_label
    relation_id_to_label = triple_tf.relation_id_to_label
    df = pd.DataFrame(
        triple_tf.mapped_triples.numpy(), columns=["source", "relation", "target"]
    )
    df["source"] = df["source"].map(entity_id_to_label)
    df["relation"] = df["relation"].map(relation_id_to_label)
    df["target"] = df["target"].map(entity_id_to_label)
    return df


def triplets_to_file(df: pd.DataFrame, data_type: str, out_folder: str | Path) -> None:
    """
    Save a DataFrame to a TSV file.

    :param df: the DataFrame to save
    :param data_type: the type of data (e.g., "train", "test", "validation")
    :param out_folder: the output directory where the TSV file will be saved

    :return: None
    """

    output_file = out_folder + "/" + data_type + ".tsv"
    df.to_csv(output_file, sep="\t", header=False, index=False)
    print(f"{data_type} saved to '{output_file}'.")
    return


def generate_drug_test_set(
    drug_class_name: str,
    kg_labels_file: str | Path,
    out_dir: str | Path,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a test set of triples for drugs only

    :param drug_class_name: the string indicating the drug class in th KG
    :param kg_labels_file: the path to the KG labels file
    :param out_dir: the output directory where the test set will be saved
    :param test_df: the DataFrame containing the test triples

    :return: a DataFrame containing the test triples for drugs only
    """

    # Assign column names to existing DataFrame
    columns = ["source", "relation", "target"]
    test_df.columns = columns

    # get a dictionary of labels
    labels = pd.read_table(f"{kg_labels_file}", dtype=str)
    labels_dict = dict(zip(labels["Name"], labels["Type"]))

    test_df["source"] = test_df["source"].astype(str)
    test_df["target"] = test_df["target"].astype(str)
    new_test_df = test_df.copy()
    new_test_df["source_type"] = new_test_df["source"].apply(labels_dict.get)
    new_test_df["target_type"] = new_test_df["target"].apply(labels_dict.get)
    filtered_df = new_test_df[
        (new_test_df["source_type"] == drug_class_name)
        & (new_test_df["target_type"] == drug_class_name)
    ]
    filtered_df = filtered_df[["source", "relation", "target"]]
    filtered_df.to_csv(
        f"{out_dir}/test_{drug_class_name}_{drug_class_name}.tsv",
        index=False,
        header=False,
        sep="\t",
    )
    return filtered_df
