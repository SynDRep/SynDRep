import json
from prepare_combos import generate_enriched_kg

generate_enriched_kg(
    kg_tsv="/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/kg.tsv",
    combos_folder="/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/combos",
    kg_drug_file="/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/drugs.csv",
    out_dir="/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data",
    name_cid_dict= json.load(open('/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/name_cid_dict.json','r'))
)

