# -*- coding: utf-8 -*-
"""wrapper for pathway analysis."""
import json

import networkx as nx
import requests


def generate_graph(kg_file: str):
    with open(kg_file, "r") as f:
        lines = f.readlines()

    # Step 2: Create Graph and Add Edges
    G = nx.DiGraph()  # You can choose a different graph type if needed

    for line in lines:
        source, relation, target = line.strip().split(
            "\t"
        )  # Assuming the format is source<TAB>target<TAB>weight
        G.add_edge(source, target, label=relation)
    return G


def get_nodes_within_n_dist(Graph, source_node, n):

    # Compute shortest path lengths from the source node
    shortest_path_lengths = nx.single_source_shortest_path_length(
        Graph, source_node, cutoff=n
    )

    # Extract nodes within a maximum of 3 hops
    nodes_within_n_dist = {
        node: distance
        for node, distance in shortest_path_lengths.items()
        if distance > 0
    }

    return nodes_within_n_dist


def get_id_or_label(query, labels_dict, required_output):
    if required_output == "name":
        return labels_dict.get(query)
    elif required_output == "ID":
        return [k for k, v in labels_dict.items() if v == query][0]


def get_shared_neighbors_within_n_dist(Graph, node1, node2, n):

    # get set of neighbors of node1
    nei_node1 = set(get_nodes_within_n_dist(Graph, node1, n).keys())

    # get set of neighbors of node2
    nei_node2 = set(get_nodes_within_n_dist(Graph, node2, n).keys())

    # get the shared neighbors
    shared_neighbors = list(nei_node1.intersection(nei_node2))
    return shared_neighbors


def get_disease_neighbors(
    Graph, drug, kg_labels, n, disease_type_str="Pathology"
):  # 'Pathology' or 'Disease':

    # get dictionary of types
    Types_dict = dict(zip(kg_labels["name"], kg_labels["Type"]))

    # get neighbors
    neighbors_type = {}
    neighbors = get_nodes_within_n_dist(Graph, drug, n)

    for neighbor, distance in neighbors.items():

        neighbors_type[neighbor] = {
            "type": Types_dict.get(neighbor),
            "distance": distance,
        }

    disease_neighbors = {
        k: v["distance"]
        for k, v in neighbors_type.items()
        if v["type"] == disease_type_str
    }

    return disease_neighbors


# getting a list of interesting diseases for a drug
def get_int_diseases(Graph, drug, kg_labels, n, disease_type_str, disease_substring):
    # get the diseases and distances
    disease_neighbors = get_disease_neighbors(
        Graph, drug, kg_labels, n, disease_type_str
    )

    # get a set of  diseases
    diseases = kg_labels[kg_labels["Type"] == disease_type_str]

    substrings = disease_substring
    mask = diseases["name"].apply(lambda x: any(sub in x.lower() for sub in substrings))
    int_diseases = diseases[mask]
    int_diseases_list = int_diseases["name"].to_list()

    int_neighbor_diseases = {
        k: v for k, v in disease_neighbors.items() if k in int_diseases_list
    }
    return int_neighbor_diseases


def get_shared_int_diseases(
    Graph, drug1, drug2, kg_labels, n, disease_type_str, disease_substring
):

    drug1_int_diseases = set(
        get_int_diseases(
            Graph, drug1, kg_labels, n, disease_type_str, disease_substring
        )
    )
    drug2_int_diseases = set(
        get_int_diseases(
            Graph, drug2, kg_labels, n, disease_type_str, disease_substring
        )
    )
    shared_int_diseases = drug1_int_diseases.intersection(drug2_int_diseases)
    return shared_int_diseases


def get_protein_neighbors(Graph, drug, kg_labels, n, protein_type_str="Protein"):

    # get dictionary of types
    Types_dict = dict(zip(kg_labels["node_id"], kg_labels["Type"]))

    # get neighbors
    neighbors_type = {}
    neighbors = get_nodes_within_n_dist(Graph, drug, n)

    for neighbor, distance in neighbors.items():

        neighbors_type[neighbor] = {
            "type": Types_dict.get(neighbor),
            "distance": distance,
        }

    protein_neighbors = {
        k: v["distance"]
        for k, v in neighbors_type.items()
        if v["type"] == protein_type_str
    }

    return protein_neighbors


def get_shared_proteins(Graph, drug1, drug2, kg_labels, n, protein_type_str):

    drug1_neighbor_protein = set(
        get_protein_neighbors(Graph, drug1, kg_labels, n, protein_type_str)
    )
    drug2_neighbor_protein = set(
        get_protein_neighbors(Graph, drug2, kg_labels, n, protein_type_str)
    )
    shared_neighbor_proteins = drug1_neighbor_protein.intersection(
        drug2_neighbor_protein
    )
    return shared_neighbor_proteins


def get_the_short_paths(Graph, node1, node2):

    # Choose two nodes
    source_node = node1
    target_node = node2

    # get the shortest paths
    shortest_paths = nx.shortest_simple_paths(
        Graph, source=source_node, target=target_node
    )
    shortest_paths_list = []
    for path in shortest_paths:
        if len(shortest_paths_list) == 0:
            shortest_paths_list.append(path)
        elif len(path) == len(shortest_paths_list[-1]):
            shortest_paths_list.append(path)
        elif len(path) > len(shortest_paths_list[-1]):
            break
    return shortest_paths_list


def only_proteins(lst, Type_dict):
    list_types = list(map(Type_dict.get, lst))
    for x in list_types[1:-1]:
        if x != "Protein":
            return False
    return True


def get_path_within_n_distance(Graph, node1, node2, n):

    # Choose two nodes
    source_node = node1
    target_node = node2

    # get the shortest paths
    paths = nx.shortest_simple_paths(Graph, source=source_node, target=target_node)
    paths_list = []
    for path in paths:
        if len(path) <= n:
            paths_list.append(path)
        else:
            break
    return paths_list


def get_shared_pathway_to_disease(Graph, node1, node2, disease, n):

    Node1_paths = get_path_within_n_distance(Graph, node1, disease, n)
    Node2_paths = get_path_within_n_distance(Graph, node2, disease, n)
    shared_paths = []
    for x in Node1_paths:
        for y in Node2_paths:
            if x[-2] == y[-2]:
                shared_paths.append((x, y))
                break
    return shared_paths


def make_complete_path(path, graph, print_path_txt=False, print_path_latex=False):

    result = []

    for i in range(len(path) - 1):
        source_node = path[i]
        target_node = path[i + 1]

        edge_data = graph.get_edge_data(source_node, target_node)
        if edge_data:
            edge_type = edge_data.get("label")
            result.append([source_node, (edge_type), target_node])
    result = [item for sublist in result for item in sublist]
    final_result = [result[0]]
    for i in range(1, len(result)):
        if result[i] == result[i - 1]:
            continue
        else:
            final_result.append(result[i])
    string = ""
    if print_path_latex and not print_path_txt:
        for i in range(len(final_result) - 1):
            if i % 2 == 0:
                string += "\\fbox{" + final_result[i].replace("_", "\_") + "}"
            else:

                string += (
                    r"$\xRightarrow{\text{" + final_result[i].replace("_", "\_") + "}}$"
                )
        string += "\\fbox{" + final_result[i + 1].replace("_", "\_") + "}"
        return final_result, string
    if print_path_txt and not print_path_latex:
        txt_print = " --> ".join(final_result)
        return final_result, txt_print
    if print_path_txt and print_path_latex:
        txt_print = " --> ".join(final_result)
        for i in range(len(final_result) - 1):
            if i % 2 == 0:
                string += "\\fbox{" + final_result[i].replace("_", "\_") + "}"
            else:
                string += (
                    r"$\xRightarrow{\text{" + final_result[i].replace("_", "\_") + "}}$"
                )
        string += "\\fbox{" + final_result[i + 1].replace("_", "\_") + "}"

        return final_result, txt_print, string
    return final_result


# TODO check the remaining functions
def get_pathways_to_all_shared_diseases(
    Graph,
    drug1,
    drug2,
    kg_labels,
    disease_type_str,
    n,
    disease_substring,
    detailed_txt=False,
    detailed_latex=False,
):
    shared_diseases = get_shared_int_diseases(
        Graph, drug1, drug2, kg_labels, n, disease_type_str, disease_substring
    )
    all_pathways = {}
    for disease in shared_diseases:
        shared_pathways = get_shared_pathway_to_disease(Graph, drug1, drug2, disease, n)
        if shared_pathways:
            all_pathways[disease] = shared_pathways
    disease_name = (
        disease_substring[0][:3] if len(disease_substring) == 1 else "all_dis"
    )
    if detailed_latex:
        if all_pathways:
            final_txt_ltx = ""
            for pathways in all_pathways.values():
                for pathway in pathways:
                    for i in range(len(pathway)):
                        d, string = make_complete_path(
                            pathway[i], Graph, print_path=False, print_path_latex=True
                        )
                        if i % 2 == 0:
                            final_txt_ltx += string + r"\\" + "\n\n"
                        else:
                            final_txt_ltx += (
                                string
                                + "\n"
                                + "\\begin{center}"
                                + "\n"
                                + r"\rule{5in}{0.4pt}\\"
                                + "\n"
                                + "\end{center}"
                                + "\n"
                            )
            with open(f"{drug1[:3]}_{drug2[:3]}_{disease_name}_ltx.txt", "w") as f:
                f.write(final_txt_ltx)
    if detailed_txt:
        if all_pathways:
            final_txt = ""
            for pathways in all_pathways.values():
                for pathway in pathways:
                    for i in range(len(pathway)):
                        d, string = make_complete_path(
                            pathway[i], Graph, print_path=True, print_path_latex=False
                        )
                        if i % 2 == 0:
                            final_txt += string + "\n"
                        else:
                            final_txt += string + "\n\n"
            with open(f"{drug1[:3]}_{drug2[:3]}_{disease_name}.txt", "w") as f:
                f.write(final_txt)
    json.dump(
        all_pathways,
        open(f"{drug1[:3]}_{drug2[:3]}_{disease_name}.json", "w"),
        indent=4,
    )
    return all_pathways


def get_otc_prescribed(disease: str) -> set:
    """
    This function retrieves a set of over-the-counter (OTC) drugs approved for a specific disease.
    It uses the OpenFDA API to search for drug labels with the specified disease in the indications and usage field.

    Parameters:
    disease (str): The disease for which to retrieve drugs.

    Returns:
    set: A set of drugs approved for the specified disease.
    """

    # Define the base URL for the OpenFDA API
    base_url = "https://api.fda.gov/drug/label.json"

    skip = 26000

    # Create a query to search for drugs approved for the specified disease
    drugs = []
    for i in range(0, skip + 1, 1000):
        print(i)
        query = f"?search=indications_and_usage:{disease}&limit=1000&skip={i}"

        # Make the API request
        response = requests.get(base_url + query)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            # Extract drug information from the API response

            for result in data["results"]:
                # if result.get('spl_product_data_elements'):
                #     drugs .append(result['spl_product_data_elements'][0].split(' ')[0])
                if result.get("openfda"):
                    if result.get("openfda").get("generic_name"):
                        drugs.append(
                            result["openfda"]["generic_name"][0].split(" ")[0].title()
                        )

        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            break

    if drugs:
        print(f"Drugs approved for {disease}:")
        for drug in set(drugs):
            print(drug)
        disease_drugs = set(drugs)

        return disease_drugs
    else:
        print(f"No drugs found for {disease}.")
        return set()
