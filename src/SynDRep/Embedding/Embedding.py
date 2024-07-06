# -*- coding: utf-8 -*-
"""do the embedding of enriched KG"""
import json
import pathlib
from itertools import combinations

import pandas as pd
import torch
from Embedding.Prediction import predict_diff_dataset
from Embedding.Prediction import predict_with_model
from pykeen.hpo.hpo import hpo_pipeline_from_path
from pykeen.pipeline import pipeline_from_config
from pykeen.triples import TriplesFactory
from Embedding.Scoring import draw_graph
from Embedding.Scoring import mean_hits
from Embedding.Scoring import multiclass_score_func
from Embedding.Scoring import percents_true_predictions
from tqdm import tqdm


def embed_and_predict(
    kg_drug_file: str,
    models_names: list,
    kg_file,
    out_dir,
    best_out_file: str = "predictions_best.csv",
    config_path=None,
    subsplits=True,
    test_specific_type=True,
    kg_labels_file=None,
    source_type="Drug",
    target_type="Drug",
    predict_all_test=False,
    all_out_file: str = None,
    with_annotation=True,
    filter_training=False,
    with_scoring=True,
    get_embeddings_data=False,
    pred_reverse=True,
    sorted_predictions=True,
    all_drug_drug_score=False,
):
    best_model = compare_embeddings(
        models_names=models_names,
        kg_file=kg_file,
        out_dir=out_dir,
        best_out_file=best_out_file,
        config_path=config_path,
        subsplits=subsplits,
        test_specific_type=test_specific_type,
        kg_labels_file=kg_labels_file,
        source_type=source_type,
        target_type=target_type,
        predict_all=predict_all_test,
        all_out_file=all_out_file,
        with_annotation=with_annotation,
        filter_training=filter_training,
        with_scoring=with_scoring,
        get_embeddings_data=get_embeddings_data
    )

    drugs = pd.read_csv(kg_drug_file)["Drug_name"].tolist()
    combs = list(combinations(drugs, 2))
    df_list_all = []
    df_list_best = []
    
    for comb in tqdm(combs):
        head, tail = comb
        pred_df = predict_with_model(
            model_name=best_model,
            out_dir=out_dir,
            subsplits=subsplits,
            drug1=head,
            drug2=tail,
        )

        # add to list of data frames
        df_list_all.append(pred_df)
        # add the best 1 to another df
        pred_df = pred_df.head(1)
        df_list_best.append(pred_df)

        if pred_reverse:
            reverse_pred_df = predict_with_model(
                model_name=best_model,
                out_dir=out_dir,
                subsplits=subsplits,
                drug1=tail,
                drug2=head,
            )

            # add to list of data frames
            df_list_all.append(reverse_pred_df)
            # add the best 1 to another df
            reverse_pred_df = reverse_pred_df.head(1)
            df_list_best.append(reverse_pred_df)

    df_all = pd.concat(df_list_all, ignore_index=True).reset_index(drop=True)
    df_best = pd.concat(df_list_best, ignore_index=True).reset_index(drop=True)

    if sorted_predictions:
        df_best.sort_values(by=["score"], ascending=False, inplace=True)

    for df in [df_all, df_best]:
        df.rename(
            columns={
                "head_label": "Drug1_name",
                "tail_label": "Drug2_name",
            },
            inplace=True,
        )
    if all_drug_drug_score:
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
    best_out_file: str = "predictions_best.csv",
    config_path=None,
    subsplits=True,
    test_specific_type=False,
    kg_labels_file=None,
    source_type=None,
    target_type=None,
    predict_all=False,
    all_out_file: str = None,
    with_annotation=True,
    filter_training=False,
    with_scoring=True,
    get_embeddings_data=False,
):

    results = []

    for model_name in models_names:
        embedding_results = kg_embedding(
            kg_file=kg_file,
            out_dir=out_dir,
            model_name=model_name,
            best_out_file=best_out_file,
            config_path=config_path,
            subsplits=subsplits,
            test_specific_type=test_specific_type,
            kg_labels_file=kg_labels_file,
            source_type=source_type,
            target_type=target_type,
            predict_all=predict_all,
            all_out_file=all_out_file,
            with_annotation=with_annotation,
            filter_training=filter_training,
            with_scoring=with_scoring,
        )
        result = [model_name]
        result.extend(embedding_results[1:])
        results.append(result)

    if test_specific_type:
        columns = [
            "model",
            "Percentage of true predictions for all relations",
            "adjusted_arithmetic_mean_rank",
            "hits_at_10",
            "roc_auc for all relations",
            f"Percentage of true predictions for {source_type}-{target_type} relations",
        ]
    else:
        columns = [
            "model",
            "Percentage of true predictions for all relations",
            "adjusted_arithmetic_mean_rank",
            "hits_at_10",
            "roc-auc for all relations",
        ]
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(f"{out_dir}/Models_results.csv", index=False)
    if test_specific_type:
        max_score_row = results_df.loc[
            results_df[
                f"Percentage of true predictions for {source_type}-{target_type} relations"
            ].idxmax()
        ]
    else:
        max_score_row = results_df.loc[
            results_df["Percentage of true predictions for all relations"].idxmax()
        ]
    best_model = max_score_row["model"]
    df = results_df.reindex(models_names)
    for metric in columns:
        df = df[metric]
        draw_graph(
            df=results_df[metric],
            title=metric.replace("_", " ") + " for all models",
            xlabel="Models",
            ylabel=metric.replace("_", " "),
            path=f"{out_dir}/{metric}.jpg",
        )
    
    if get_embeddings_data:
        get_embeddings(
            model=torch.load(f"{out_dir}/{best_model}/{best_model}_best_model_results/trained_model.pkl"),
            training_tf=embedding_results[0],
            kg_labels_file=kg_labels_file,
            output_all=f'{out_dir}/{best_model}/{best_model}_embeddings_all.csv',
            output_drugs=f'{out_dir}/{best_model}/{best_model}_embeddings_drugs.csv',
        )
        
    return best_model


def kg_embedding(
    kg_file,
    out_dir,
    model_name,
    best_out_file: str = "predictions_best.csv",
    config_path=None,
    subsplits=True,
    test_specific_type=False,
    kg_labels_file=None,
    source_type=None,
    target_type=None,
    predict_all=False,
    all_out_file: str = None,
    with_annotation=True,
    filter_training=False,
    with_scoring=True,
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
            subsplits,
            test_specific_type,
            kg_labels_file,
            source_type,
            target_type,
        )
    else:
        train_df, test_df, validation_df, train_tf, test_tf, validation_tf = (
            create_data_splits(
                kg_file,
                out_dir,
                subsplits,
                test_specific_type,
                kg_labels_file,
                source_type,
                target_type,
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
        with_annotation=with_annotation,
        subsplits=subsplits,
        training_df=train_df,
        testing_df=test_df,
        validation_df=validation_df,
        filter_training=filter_training,
        predict_all=predict_all,
        all_out_file=all_out_file,
    )
    # scoring
    if with_scoring:
        best_test_pred = pd.read_csv(f"{out_dir}/{model_name}/{best_out_file}")
        percent_true = percents_true_predictions(best_test_pred)
        mean, hits = mean_hits(
            f"{out_dir}/{model_name}/{model_name}_best_model_results/results.json"
        )
        roc = multiclass_score_func(best_test_pred, main_test_df)

    # specif types testing set predictions
    if test_specific_type:
        sp_test_df = pd.read_table(
            f"{out_dir}/test_{source_type}_{target_type}.tsv",
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
            best_out_file=f"{source_type}_{target_type}_{best_out_file}",
            with_annotation=with_annotation,
            subsplits=subsplits,
            training_df=train_df,
            testing_df=test_df,
            validation_df=validation_df,
            filter_training=filter_training,
            predict_all=predict_all,
            all_out_file=f"{source_type}_{target_type}_{all_out_file}",
        )
        if with_scoring:
            best_test_pred_sp_type = pd.read_csv(
                f"{out_dir}/{model_name}/{source_type}_{target_type}_{best_out_file}"
            )
            percent_true_sp_type = percents_true_predictions(best_test_pred_sp_type)

            
            results_dict = {
                "Percentage of true predictions for all relations": percent_true,
                "roc_auc for all relations": roc,
                "adjusted_arithmetic_mean_rank": mean,
                "hits_at_10": hits,
                f"Percentage of true predictions for {source_type}-{target_type} relations": percent_true_sp_type,
                
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
                percent_true,
                mean,
                hits,
                roc,
                percent_true_sp_type,
            )

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
    return (train_tf, percent_true, mean, hits, roc)

def get_embeddings(
    model,
    training_tf,
    kg_labels_file,
    output_all,
    output_drugs
):

    #get embeddings
    
    entity_embeddings_array = model.entity_representations[0](indices=None).detach().cpu().numpy().real
    entities = training_tf.entity_id_to_label
    entity_ids = range(model.num_entities)
    embedding_columns = [f'embedding_{i}' for i in range(entity_embeddings_array.shape[1])]
    data = {col: entity_embeddings_array[:, i] for i, col in enumerate(embedding_columns)}
    data['entity'] = [entities[entity_id] for entity_id in entity_ids]
    embeddings_df = pd.DataFrame(data)

    # adding labels
    labels = pd.read_table(kg_labels_file, dtype=str)
    labels_dict = dict (zip(labels['ID'],labels["Type"]))
    names_dict = dict (zip(labels['ID'],labels["name"]))
    embeddings_df['entity'] = embeddings_df['entity'].astype(str)
    embeddings_df ["entity_type"]= embeddings_df['entity'].apply(labels_dict.get)
    embeddings_df ["entity_name"]= embeddings_df['entity'].apply(names_dict.get)
    embeddings_df = embeddings_df[['entity', 'entity_type', 'entity_name'] + embedding_columns]
    embeddings_df.to_csv(output_all, index=False)

    drug_embeddings = embeddings_df[embeddings_df['entity_type'] == "Drug" ]
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
    subsplits=True,
    test_specific_type=False,
    kg_labels_file=None,
    source_type=None,
    target_type=None,
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
        if test_specific_type:
            if all([kg_labels_file, source_type, target_type]):
                generate_test_specific_type(
                    kg_labels_file, main_test_df, source_type, target_type, out_dir
                )
            else:
                raise TypeError(
                    "One or many of kg_labels_file, test_df, source_type, target_type is/are not given"
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

    if test_specific_type:
        if all([kg_labels_file, source_type, target_type]):
            generate_test_specific_type(
                kg_labels_file, test_df, source_type, target_type, out_dir
            )
        else:
            raise TypeError(
                "One or many of kg_labels_file, test_df, source_type, target_type is/are not given"
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


def generate_test_specific_type(
    kg_labels_file, test_df, source_type, target_type, out_dir
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
        (new_test_df["source_type"] == source_type)
        & (new_test_df["target_type"] == target_type)
    ]
    filtered_df = filtered_df[["source", "relation", "target"]]
    filtered_df.to_csv(
        f"{out_dir}/test_{source_type}_{target_type}.tsv",
        index=False,
        header=False,
        sep="\t",
    )
    return filtered_df
