# -*- coding: utf-8 -*-
"""Analysis od drug-drug combination interactions."""

import pandas as pd
from tqdm import tqdm


def generate_relation_specific_df(
    df:pd.DataFrame,
    relation_type: str,
    out_dir: str,
    reverse_predict: bool = False,
    ) -> None:
    
    

    selected_df = df[(df["relation_label"] == relation_type)]
    selected_df = selected_df.sort_values(by=["score"], ascending=False)
    
    if reverse_predict:
    # get unique drug pairs
        unique_pairs = set(zip(selected_df['Drug1_name'], selected_df['Drug2_name']))
        filtered_rows = []
        processed_pairs = []
        for pair in tqdm(unique_pairs):
            # add the reverse of the current pair to processed_pairs to avoid duplication
            reverse_pair = (pair[1], pair[0])
            # when processing the reverse pair it should stop as it is already processed with the original pair
            if reverse_pair in processed_pairs:
                continue
            # if both the pred and reverse pred are synergistic  
            elif reverse_pair in unique_pairs:
                filtered_rows.append(selected_df[(selected_df['Drug1_name'] == pair[0]) & (selected_df['Drug2_name'] == pair[1])])
                #add pair to processed
                processed_pairs.append(pair)
        final_df = pd.concat(filtered_rows).reset_index(drop=True)
        final_df = final_df.sort_values(by=["score"], ascending=False)
        final_df.to_csv(f"{out_dir}/{relation_type}_pred.csv", index=False)
        return final_df
    else:
        final_df = selected_df.reset_index(drop=True)
        final_df.to_csv(f"{out_dir}/{relation_type}_pred.csv", index=False)
        return final_df
    

