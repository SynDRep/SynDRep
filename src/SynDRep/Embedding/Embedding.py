# -*- coding: utf-8 -*-

"""do the embedding of enriched KG"""


import pandas as pd
from pykeen.triples import TriplesFactory


def create_data_splits(
    kg_file,
    data_out_dir,
    subsplits=True,
    test_specific_type=False,
    kg_labels_file=None,
    source_type=None,
    target_type=None,
):
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

    for data in ["train_data", "test_data", "validation_data"]:
        triplets_to_file(data_out_dir, eval(data.replace("data", "df")), data)
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
            train_df, test_df, validation_df, train_tf, test_tf, validation_tf = (
                make_splits(train_tf)
            )

        print(
            "All nodes and relations in test and validation subsets are there in the training set :)"
        )
        for data in [
            "train_data_ss",
            "test_data_ss",
            "validation_data_ss",
            "main_test_data",
        ]:
            triplets_to_file(data_out_dir, eval(data.replace("data", "df")), data)

        print("all done :)")
        if test_specific_type:
            if all(kg_labels_file, test_df, source_type, target_type):
                generate_test_specific_type(
                    kg_labels_file, main_test_df, source_type, target_type, data_out_dir
                )
            else:
                raise TypeError(
                    "One or many of kg_labels_file, test_df, source_type, target_type is/are not given"
                )

        return train_tf_ss, test_tf_ss, validation_tf_ss

    if test_specific_type:
        if all(kg_labels_file, test_df, source_type, target_type):
            generate_test_specific_type(
                kg_labels_file, test_df, source_type, target_type, data_out_dir
            )
        else:
            raise TypeError(
                "One or many of kg_labels_file, test_df, source_type, target_type is/are not given"
            )

    print("all done :)")
    return train_tf, test_tf, validation_tf


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


def triples_to_df(tripl_tf):
    df = pd.DataFrame(
        tripl_tf.mapped_triples.numpy(), columns=["source", "relation", "target"]
    )
    return df


def triplets_to_file(out_folder: str, df: pd.DataFrame, data_type):

    output_file = out_folder + "/" + data_type + ".tsv"
    df.to_csv(output_file, sep="\t", header=False, index=False)
    print(f"Train triplets saved to '{output_file}'.")
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
    test_df["source_type"] = test_df["source"].apply(labels_dict.get)
    test_df["target_type"] = test_df["target"].apply(labels_dict.get)
    filtered_df = test_df[
        (test_df["source_type"] == source_type)
        & (test_df["target_type"] == target_type)
    ]
    filtered_df = filtered_df[["source", "relation", "target"]]
    filtered_df.to_csv(
        f"{out_dir}/test_{source_type}_{target_type}.tsv",
        index=False,
        header=False,
        sep="\t",
    )
    return filtered_df
