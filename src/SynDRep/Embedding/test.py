from Embedding import create_data_splits


create_data_splits(
    kg_file='/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/enriched_kg.tsv',
    out_dir='/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data',
    subsplits=False,
    test_specific_type=True,
    source_type='Gene',
    target_type='Protein',
    kg_labels_file='/home/kshalaby/Nextcloud/SynDRep/src/SynDRep/test_data/kg_labels.tsv'
    
)