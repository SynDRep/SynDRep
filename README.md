



<p align="center">
  <img src="docs/source/logo.jpg">
</p>

<h1 align="center">
  SynDRep: A Knowledge Graph-Enhanced Tool based on Synergistic Partner Prediction for Drug Repurposing
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
$ pip install git+https://github.com/SynDRep/SynDRep.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/syndrep) and installed in editable mode with:

```bash
$ git clone https://github.com/SynDRep/SynDRep.git
$ cd SynDRep
$ pip install -e .
```

## Documentation



## Input Data Formats
The following four files are required to run most of the SynDRep functions in the repository:

### 1. Drug combinations

| Drug1_name    | Drug1_CID | Drug2_name | Drug2_CID | HSA | Bliss | Loewe | ZIP | Source_DB | DB_ID
| --------- | -------- | -------- | -------- |  --------- | -------- | -------- | -------- | --------- | -------- | 
| gemcitabine | 60750 | Lapatinib | 208908 | 8.89 | 8.33 | 2.1 | 10.98 | DrugcombPortal | 48775 |
| GEFITINIB | 123631 | THIOGUANINE | 2723601 | 3.03 | -2.67 | 1.01 | -8.43 | DrugcombDB | 168014 |
|Crizotinib | 11626560 | MITHRAMYCIN | 163659 | 4.23 | 5.73 | 4.46 | -0.33 | DrugcombPortal | 355424 |

**Note:** The data must be in a comma separated file format. Combination data must be kept in a separate folder containing only the CSV files of combinations. Combinations from different databases can be added to this folder as separate CSV files and they will be merged together.

### 2. Knowledge Graph

The graph format should be a modified version of the edge List Format. Which looks as follows:

| Source    | Relation    | Target    |
| --------- | ----------- | --------- |
| p(HGNC:TARDBP) | INCREASES | r(HGNC:HDAC6)
| p(MGI:Gnaq) | KEGG_INCREASES | p(MGI:Pla2g2c)
| Estramustine | EBEL_DRUGBANK_RELATION | p(HGNC:ESR1)
    
**Note:** The data must be in a tab separated file format and **without headers**.

### 3. Knowledge Graph Labels

a file containing the nodes Name and types such as protein, gene, or drug.

| Type    | Name    |
| --------- | ----------- |
| BiologicalProcess | bp(MESHPP:Neuroprotection) |
| Drug | Pegfilgrastim
|Protein | p(HGNC:CPEB2)
    
**Note:** The data must be in a tab separated file format.

### 4. Knowledge Graph Drugs

a file containing the drugs in KG.  It should be a CSV file containing all the names of drugs in KG. It shoud have "Drug_name" column and any other columns.
    

## Usage

**Note:** These are very basic commands for SynDRep:

1. **Run the SynDRep tool**
The following command generates predicted synergy based on the method chosen.

```bash
$ SynDRep run-syndrep -bof <BEST_OUT_FILE> -cf <COMBOS_FOLDER> -cp <CONFIG_PATH> -d <DEVICE> -dcn <DRUG_CLASS_NAME> -emn <EM_MODELS_NAME> -ft <FILTER_TRAINING_SET> -kdf <KG_DRUG_FILE> -kf <KG_FILE> -klf <KG_LABELS_FILE> -m <METHOD_FOR_SYNDREP> -mmn <ML_MODELS_NAME> -nb <NUMBER_OF_BITS_FOR_MORGAN_FP> -ncd <NAME_CID_DICTIONARY>  -on <OPTIMIZER_NAME> -o <OUTPUT_DIR> -pr <PREDICT_REVERSE_RELATION_BETWEEN_DRUGS:BOOL> -r <RADIUS_FOR_MORGAN_FP> -rl <RANDOM_LABELS_FOR_ML> -sys <SYNERGY_SCORING_MODEL> -sml <SCORING_METRIC_FOR_ML> -s <MAKE_SUBSPLITS> -vcv <VALIDATION_CV_FOR_ML>
```

2. **Enriched Kg generation**
The following command generates a KG enriched with drug-drug combinations from KG, drugs, and combination files.

```bash
$ SynDRep enriched-kg -o <OUTPUT_DIR> -kf <KG_FILE> --kdf <KG_DRUG_FILE> -ncd <NAME_CID_DICTIONARY> -cf <COMBOS_FOLDER> 
```

3. **Knowledge Graph Embedding only**

The following command generates the embedding of the knowledge graph passed to it along with presiction of relation between the drugs in this KG.

```bash
$ SynDRep embedding-and-prediction -bof <BEST_OUT_FILE> cp <CONFIG_PATH> -dcn <DRUG_CLASS_NAME> -emn <EM_MODELS_NAME> --ft <FILTER_TRAINING_SET> -ged <GET_EMBEDDINGS:BOOL> -kdf <KG_DRUG_FILE> -kf <KG_FILE> -klf <KG_LABELS_FILE> -o <OUTPUT_DIR> -pr <PREDICT_REVERSE_RELATION_BETWEEN_DRUGS:BOOL>  s <MAKE_SUBSPLITS> 
```

4. **Classification**

The following command carries out classification on the given data file for training using chosen models, and produce prediction for the data file for prediction.

```bash
$ SynDRep classify -dt <DATA_FOR_TRAINING_CSV_FILE> -dp <DATA_FOR_PREDICTION_CSV_FILE> -mmn <ML_MODELS_NAME> -on <OPTIMIZER_NAME> -o <OUTPUT_DIR> -rl <RANDOM_LABELS_FOR_ML> -sml <SCORING_METRIC_FOR_ML> -vcv <VALIDATION_CV_FOR_ML>
```

## Issues

If you have difficulties using SynDRep, please open an issue at our [GitHub](https://github.com/syndrep) repository.

## Acknowledgements

### Citation



### Graphics



## Disclaimer

SynDRep is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or guarantee of maintenance, support, or back-up of data.
