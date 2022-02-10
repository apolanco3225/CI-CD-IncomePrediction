"""
Script to validate machine learning model.
Author: Arturo Polanco
Date: February 2022
"""

# import necessary packages
from joblib import load
import os
from .ml.model import scores_in_slices

def validate_model(test_data, categorical_features, output_path):
    # define paths
    model_path = os.path.join(output_path, "model.joblib")
    encoder_path = os.path.join(output_path, "encoder.joblib")
    label_binarizer_path = os.path.join(output_path, "label_binarizer.joblib")
    
    model = load(model_path)
    encoder = load(encoder_path)
    label_binarizer = load(label_binarizer_path)

    scores_in_slices(model, test_data, encoder, label_binarizer, categorical_features, output_path)
