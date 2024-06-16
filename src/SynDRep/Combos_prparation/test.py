import json
from prepare_combos import prepare_combinations

prepare_combinations(
    combos_folder="/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/combos",
    kg_drug_file="/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/drugs.csv",
    out_dir="/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data",
    #name_cid_dict= json.load(open('/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/name_cid_dict.json','r'))
)