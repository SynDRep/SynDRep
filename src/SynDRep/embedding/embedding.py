# -*- coding: utf-8 -*-
"""do the embedding of enriched KG"""
import json
import pathlib
from itertools import combinations

import pandas as pd
import torch
from pykeen.hpo.hpo import hpo_pipeline_from_path
from pykeen.pipeline import pipeline_from_config
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
from tqdm import tqdm

from .prediction import gz_to_dict
from .prediction import predict_diff_dataset
from .scoring import draw_graph
from .scoring import mean_hits
from .scoring import multiclass_score_func
from .scoring import percents_true_predictions


def embed_and_predict(
    kg_drug_file,
    models_names: list,
    kg_file,
    out_dir,
    kg_labels_file,
    drug_class_name,
    all_drug_drug_predictions=False,
    all_out_file: str = None,
    best_out_file: str = "predictions_best.csv",
    config_path=None,
    filter_training=False,
    get_embeddings_data=False,
    pred_reverse=True,
    predict_all_test=False,
    sorted_predictions=True,
    subsplits=True,
):
    best_model = compare_embeddings(
        models_names=models_names,
        kg_file=kg_file,
        out_dir=out_dir,
        drug_class_name=drug_class_name,
        best_out_file=best_out_file,
        config_path=config_path,
        subsplits=subsplits,
        kg_labels_file=kg_labels_file,
        predict_all=predict_all_test,
        all_out_file=all_out_file,
        filter_training=filter_training,
        get_embeddings_data=get_embeddings_data,
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
            reverse_pred_df_best= reverse_pred_df.head(1)
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
    models_names: list,
    kg_file,
    out_dir,
    kg_labels_file,
    drug_class_name,
    best_out_file: str = "predictions_best.csv",
    config_path=None,
    subsplits=True,
    predict_all=False,
    all_out_file: str = None,
    filter_training=False,
    get_embeddings_data=False,
):

    results = []

    for model_name in models_names:
        embedding_results = kg_embedding(
            kg_file=kg_file,
            out_dir=out_dir,
            model_name=model_name,
            drug_class_name=drug_class_name,
            best_out_file=best_out_file,
            config_path=config_path,
            subsplits=subsplits,
            kg_labels_file=kg_labels_file,
            predict_all=predict_all,
            all_out_file=all_out_file,
            filter_training=filter_training,
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
    max_score_row = results_df.loc[
        results_df[
            f"Percentage of true predictions for {drug_class_name}-{drug_class_name} relations"
        ].idxmax()
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
    kg_file,
    out_dir,
    model_name,
    kg_labels_file,
    drug_class_name,
    best_out_file: str = "predictions_best.csv",
    config_path=None,
    subsplits=True,
    predict_all=False,
    all_out_file: str = None,
    filter_training=False,
):

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
        kg_file,
        out_dir,
        kg_labels_file,
        drug_class_name,
        subsplits=True,
    )
    else:
        train_df, test_df, validation_df, train_tf, test_tf, validation_tf = (
            create_data_splits(
            kg_file,
            out_dir,
            kg_labels_file,
            drug_class_name,
            subsplits=False,
        )
        )

        main_test_df = pd.concat([test_df, validation_df], ignore_index=True)

    if config_path:
        config_file = config_path
    else:
        config_file = f"{model_name}_config_hpo.json"

    # run hpo
    run_hpo(config_file, train_tf, test_tf, validation_tf, model_name, out_dir)

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


def get_embeddings(model, training_tf, kg_labels_file, output_all, output_drugs):

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
    name_type_dict = dict(zip(labels["name"], labels["Type"]))

    embeddings_df["entity"] = embeddings_df["entity"].astype(str)
    embeddings_df["entity_type"] = embeddings_df["entity"].apply(name_type_dict.get)
    embeddings_df = embeddings_df[["entity", "entity_type"] + embedding_columns]
    embeddings_df.to_csv(output_all, index=False)

    drug_embeddings = embeddings_df[embeddings_df["entity_type"] == "Drug"]
    drug_embeddings.to_csv(output_drugs, index=False)
    return


def run_pipeline(train_tf, test_tf, validation_tf, model_name, out_dir):
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


def run_hpo(config_file, train_tf, test_tf, validation_tf, model_name, out_dir):
    hpo_results = hpo_pipeline_from_path(
        path=config_file, training=train_tf, validation=validation_tf, testing=test_tf
    )
    pathlib.Path(f"{out_dir}/{model_name}").mkdir(parents=True, exist_ok=True)
    hpo_results.save_to_directory(f"{out_dir}/{model_name}/{model_name}_hpo_results")
    return hpo_results


def create_data_splits(
    kg_file,
    out_dir,
    kg_labels_file,
    drug_class_name,
    subsplits=True,
):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Create a TriplesFactory object from your knowledge graph file
    kg_triples_factory = TriplesFactory.from_path(kg_file)
    train_df, test_df, validation_df, train_tf, test_tf, validation_tf = make_splits(
        kg_triples_factory
    )

    # Split the knowledge graph into train, validation, and test sets
    while not nodes_relations_check(train_df, test_df, validation_df):
        train_df, test_df, validation_df, train_tf, test_tf, validation_tf = (
            make_splits(kg_triples_factory)
        )

    print(
        "All nodes and relations in test and validation sets are there in the training set :)"
    )

    if subsplits:
        main_test_df = pd.concat([test_df, validation_df], ignore_index=True)
        (
            train_df_ss,
            test_df_ss,
            validation_df_ss,
            train_tf_ss,
            test_tf_ss,
            validation_tf_ss,
        ) = make_splits(train_tf)
        while not nodes_relations_check(train_df_ss, test_df_ss, validation_df_ss):
            (
                train_df_ss,
                test_df_ss,
                validation_df_ss,
                train_tf_ss,
                test_tf_ss,
                validation_tf_ss,
            ) = make_splits(train_tf)

        print(
            "All nodes and relations in test and validation subsets are there in the training set :)"
        )
        for data in [
            "train_data_ss",
            "test_data_ss",
            "validation_data_ss",
            "main_test_data",
        ]:
            triplets_to_file(out_dir, eval(data.replace("data", "df")), data)

        print("all done :)")
        generate_drug_test_set(
            kg_labels_file, test_df, drug_class_name, out_dir
        )

        return (
            train_df_ss,
            test_df_ss,
            validation_df_ss,
            train_tf_ss,
            test_tf_ss,
            validation_tf_ss,
            main_test_df,
        )

    
    generate_drug_test_set(
        kg_labels_file, test_df, drug_class_name, out_dir
    )
        
    for data in ["train_data", "test_data", "validation_data"]:
        triplets_to_file(out_dir, eval(data.replace("data", "df")), data)
    print("all done :)")
    return train_df, test_df, validation_df, train_tf, test_tf, validation_tf


def make_splits(kg_triples_factory):
    train_tf, test_tf, validation_tf = kg_triples_factory.split(
        [0.8, 0.1, 0.1], random_state=42
    )
    train_df = triples_to_df(train_tf)
    test_df = triples_to_df(test_tf)
    validation_df = triples_to_df(validation_tf)

    return train_df, test_df, validation_df, train_tf, test_tf, validation_tf


# check all nodes and relations in test and validation are in training
def nodes_relations_check(train_df, test_df, validation_df):

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


def get_nodes_and_relations(df):

    return set(df["source"]).union(df["target"]), set(df["relation"])


def triples_to_df(triple_tf):
    entity_id_to_label = triple_tf.entity_id_to_label
    relation_id_to_label = triple_tf.relation_id_to_label
    df = pd.DataFrame(
        triple_tf.mapped_triples.numpy(), columns=["source", "relation", "target"]
    )
    df["source"] = df["source"].map(entity_id_to_label)
    df["relation"] = df["relation"].map(relation_id_to_label)
    df["target"] = df["target"].map(entity_id_to_label)
    return df


def triplets_to_file(out_folder: str, df: pd.DataFrame, data_type):

    output_file = out_folder + "/" + data_type + ".tsv"
    df.to_csv(output_file, sep="\t", header=False, index=False)
    print(f"{data_type} saved to '{output_file}'.")
    return


def generate_drug_test_set(
    kg_labels_file, test_df, drug_class_name, out_dir
):

    # Assign column names to existing DataFrame
    columns = ["source", "relation", "target"]
    test_df.columns = columns

    # get a dictionary of labels
    labels = pd.read_table(f"{kg_labels_file}", dtype=str)
    labels_dict = dict(zip(labels["name"], labels["Type"]))

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
