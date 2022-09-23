import argparse
import os

import gcsfs
import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X_feature_names = [
    'appeal', 'facts', 'want to buy it',
    'overstatement', 'fomo', 'word_count',
    'avg_word_length', 'punct_signs_count', 'title_word_count',
    'stopword_count', 'polarity', 'subjectivity',
    'verb_count', 'adj_count', 'adv_count', 'pron_count', 'brisque_score'
]
target_col_name = 'add_to_cart'

def get_x_and_y(df, features_list, target_col_name):
    return df[features_list], df[target_col_name]

def make_dataset(client_name, data_date):
    fs = gcsfs.GCSFileSystem()
    
    data_path = f'gbi_ml/gbi_persuasion/{client_name}/{data_date}/lingustic_features_extracted.csv'
    with fs.open(data_path, 'rb') as f:
        df_linguistic = pd.read_csv(f, index_col=0)
        
    data_path = f'gbi_ml/gbi_persuasion/{client_name}/{data_date}/image_processed.csv'
    with fs.open(data_path, 'rb') as f:
        df_image = pd.read_csv(f, index_col=0)
        
    df = pd.concat([df_linguistic, df_image], axis=1)
    df.fillna(0, inplace=True)
    train_ids, test_ids = train_test_split(df.index, test_size=0.1, random_state=42)
    X_train, y_train = get_x_and_y(df.loc[train_ids], features_list=X_feature_names, target_col_name=target_col_name)
    X_test, y_test = get_x_and_y(df.loc[test_ids], features_list=X_feature_names, target_col_name=target_col_name)
    return (X_train, y_train), (X_test, y_test)

def build_model(n_estimators, max_depth, min_samples_split, learning_rate):
    """Build model for shopper persuasion prediction."""
    model_params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'learning_rate': learning_rate,
        'loss': 'squared_error',
        'random_state': 42
    }
    print(model_params)
    regressor = GradientBoostingRegressor(**model_params)
    model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    return pipeline