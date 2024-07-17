
import pandas as pd
from SynDRep.ML.classify import classify_data


data_for_training = pd.read_csv('../test_data/all_train_test_data.csv')
data_for_prediction = pd.read_csv('../test_data/pred_dataset_ML.csv')
model_names=['random_forest', 'elastic_net']
scoring_metrics = ["accuracy", "roc_auc","f1_weighted", "f1", ]
classify_data(
    data_for_training=data_for_training,
    data_for_prediction=data_for_prediction,
    optimizer_name='grid_search',
    model_names=model_names,
    out_dir='../test_data/test_output',
    validation_cv=5,
    scoring_metrics=scoring_metrics,
    rand_labels=False,
    )