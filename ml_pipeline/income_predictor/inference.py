"""
Script to make inference with machine learning model.
Author: Arturo Polanco
Date: February 2022
"""
# import necessary libraries    
from joblib import load
from .ml.data import process_data
from .ml.model import inference
import os
    



def run_inference(data, cat_features, artifacts_path):
    """
    Load model and run inference

    Parameters
    ----------
        data
        cat_features
        artifacts_path

    prediction
    -------
        
    """

    # define reading paths
    model_path = os.path.join(artifacts_path, "model.joblib")
    encoder_path = os.path.join(artifacts_path, "encoder.joblib")
    label_binarizer_path = os.path.join(artifacts_path, "label_binarizer.joblib")

    model = load(model_path)
    encoder = load(encoder_path)
    label_binarizer = load(label_binarizer_path)

    
    input_data, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder, 
        lb=label_binarizer, 
        training=False)
    

    pred = inference(model, input_data)
    prediction = label_binarizer.inverse_transform(pred)[0]

    return prediction