"""
Script to train machine learning model.
Author: Arturo Polanco
Date: February 2022
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump
import os
from .ml.data import process_data
from .ml.model import train_model

def split_data(data_path, test_size=0.20):
    """
    
    """
    dataset = pd.read_csv(data_path)
    train_data, test_data = train_test_split(dataset, test_size=test_size)
    return train_data, test_data



def train_save_model(data, output_path, categorical_features=None):
    """
    Training income classifier and save model
    """


    if categorical_features is None:
        categorical_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
    train_features, train_label, encoder, label_binarizer = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )

    model = train_model(train_features, train_label)

    # define output paths
    model_path = os.path.join(output_path, "model.joblib")
    encoder_path = os.path.join(output_path, "encoder.joblib")
    label_binarizer_path = os.path.join(output_path, "label_binarizer.joblib")

    # save output artifacts
    dump(model, model_path)
    dump(encoder, encoder_path)
    dump(label_binarizer, label_binarizer_path)
