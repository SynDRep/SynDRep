



<p align="center">
  <img src="docs/source/logo.jpg">
</p>

<h1 align="center">
  SynDRep: A Knowledge Graph-Enhanced Tool based on Synergistic Partner Prediction for Drug Repurposing
  <!-- <br/>
  <a href='https://travis-ci.com/github/hybrid-kg'>
     <img src="https://travis-ci.com/hybrid-kg/clep.svg?branch=master" />
  </a>
  <a href='https://clep.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/clep/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a href="https://zenodo.org/badge/latestdoi/209278408">
    <img src="https://zenodo.org/badge/209278408.svg" alt="DOI">
  </a>
  <a href="https://pypi.org/project/clep/">
    <img src="https://img.shields.io/pypi/v/clep" alt="CLEP on PyPI">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/clep" alt="CLEP Python versions">
  <a href="https://github.com/hybrid-kg/clep/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/clep" alt="CLEP Software License">
  </a> -->
</h1>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [General Info](#general-info)
- [Installation](#installation)
- [Documentation](#documentation)
- [Input Data Formats](#input-data-formats)
  - [Drug combinations](#drug-combinations)
  - [Knowledge Graph](#knowledge-graph)
  - [Knowledge Graph Labels](#knowledge-graph-labels)
  - [Knowledge Graph Drugs](#knowledge-graph-drugs)

- [Usage](#usage)
- [Issues](#issues)
- [Acknowledgements](#acknowledgements)
  - [Citation](#citation)
  - [Graphics](#graphics)
- [Disclaimer](#disclaimer)

## General Info

SynDRep is  a novel drug repurposing tool based on enriching knowledge graphs with drug combination effects, then identifing the repurposing candidates the by prediction of synergistic drug partners for drugs commonly prescribed for target disease. it allows for the confirmation of existence of pathways  between the candidate drug and the target disease within the knowledge graph.

## Installation


The most recent code can be installed from the source on [GitHub](https://github.com/syndrep) with:

```bash
$ pip install git+https://github.com/syndrep.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/syndrep) and installed in editable mode with:

```bash
$ git clone https://github.com/syndrep.git
$ cd syndrep
$ pip install -e .
```

## Documentation

<!-- Read the [official docs](https://clep.readthedocs.io/en/latest/) for more information. -->

## Input Data Formats

### Drug combinations

| Drug1_name    | Drug1_CID | Drug2_name | Drug2_CID | HSA | Bliss | Loewe | ZIP | Source_DB | DB_ID
| --------- | -------- | -------- | -------- |  --------- | -------- | -------- | -------- | --------- | -------- | 
| gemcitabine | 60750 | Lapatinib | 208908 | 8.89 | 8.33 | 2.1 | 10.98 | DrugcombPortal | 48775 |
| GEFITINIB | 123631 | THIOGUANINE | 2723601 | 3.03 | -2.67 | 1.01 | -8.43 | DrugcombDB | 168014 |
|Crizotinib | 11626560 | MITHRAMYCIN | 163659 | 4.23 | 5.73 | 4.46 | -0.33 | DrugcombPortal | 355424 |

**Note:** The data must be in a comma separated file format. Combination data must be kept in a separate folder containing only the CSV files of combinations. Combinations from different databases can be added to this folder as separate files and they will be merged together.

### Knowledge Graph

The graph format should be a modified version of the Edge List Format. Which looks as follows:

| Source    | Relation    | Target    |
| --------- | ----------- | --------- |
| p(HGNC:TARDBP) | INCREASES | r(HGNC:HDAC6)
| p(MGI:Gnaq) | KEGG_INCREASES | p(MGI:Pla2g2c)
| Estramustine | EBEL_DRUGBANK_RELATION | p(HGNC:ESR1)
    
**Note:** The data must be in a tab separated file format.

### Knowledge Graph Labels

a file containing the nodes Name and types such as protein, gene, or drug.

| Type    | Name    |
| --------- | ----------- |
| BiologicalProcess | bp(MESHPP:Neuroprotection) |
| Drug | Pegfilgrastim
|Protein | p(HGNC:CPEB2)
    
**Note:** The data must be in a tab separated file format.

### Knowledge Graph Drugs

a file containing the drugs in KG.  It should be a CSV file containing all the names of drugs in KG. It shoud have "Drug_name" column and any other columns.
    

## Usage

**Note:** These are very basic commands for SynDRep, and the detailed options for each command can be found in the [documentation](#documentation)

1. **Enriched Kg generation**
The following command generates a KG enriched with drug-drug combinations from KG, drugs, and combination files.

```bash
$ syndrep enriched-kg -o <OUTPUT_DIR> -k <KG_TSV_FILE> -d <KG_DRUGS_CSV_FILE> -n <NAME_CID_DICTIONARY> -c <COMBINATIONS_FOLDER>
```
<!-- 
2. **Graph Generation**
The following command generates the patient-gene network based on the method chosen (Interaction_network).

```bash
$ clep embedding generate-network --data <SCORED_DATA_FILE> --method interaction_network --ret_summary --out <OUTPUT_DIR>
```

3. **Knowledge Graph Embedding**

The following command generates the embedding of the network passed to it.

```bash
$ clep embedding kge --data <NETWORK_FILE> --design <DESIGN_FILE> --model_config <MODEL_CONFIG.json> --train_size 0.8 --validation_size 0.1 --out <OUTPUT_DIR>
```

4. **Classification**

The following command carries out classification on the given data file for a chosen model (Elastic Net) with 100 hyper-parameter optimization trials.

```bash
$ clep classify --data <EMBEDDING_FILE> --model elastic_net --num-trials 100 --out <OUTPUT_DIR>
``` -->

## Issues

If you have difficulties using SynDRep, please open an issue at our [GitHub](https://github.com/syndrep) repository.

## Acknowledgements

### Citation

<!-- If you have found CLEP useful in your work, please consider citing:

[**CLEP: A Hybrid Data- and Knowledge- Driven Framework for Generating Patient Representations**](https://doi.org/10.1093/bioinformatics/btab340
).<br />
Bharadhwaj, V. S., Ali, M., Birkenbihl, C., Mubeen, S., Lehmann, J., Hofmann-Apitius, M., Hoyt, C. T., & Domingo-Fernandez, D. (2020).<br />
*Bioinformatics*, btab340.  -->

### Graphics

<!-- The CLEP logo and framework graphic was designed by Carina Steinborn. -->

## Disclaimer

SynDRep is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or guarantee of maintenance, support, or back-up of data.
