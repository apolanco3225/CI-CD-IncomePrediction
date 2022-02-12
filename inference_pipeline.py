
import os 
import yaml
from joblib import load
import pandas as pd
import numpy as np

def inference(sample):
    """
    Inference machine learning
    """

    # importing values from config file
    with open('ml_pipeline/config.yml', encoding='utf8') as file:
        config = yaml.safe_load(file)

    cat_features = config["data"]["cat_features"]
    artifacts_path = "ml_pipeline/model"

    # define reading paths
    model_path = os.path.join(artifacts_path, "model.joblib")
    encoder_path = os.path.join(artifacts_path, "encoder.joblib")
    label_binarizer_path = os.path.join(
        artifacts_path, "label_binarizer.joblib")

    # loading model, encoder and label binarizer
    model = load(model_path)
    encoder = load(encoder_path)
    label_binarizer = load(label_binarizer_path)
    cat_features = config['data']['cat_features']

    # converting json file into dictionary
    sample = sample.dict()
    # conver dictionary into a pandas dataframe
    data = pd.DataFrame(data=sample.values(), index=sample.keys()).T

    # data preprocessing
    categorical_features = data[cat_features].values
    continuous_features = data.drop(*[cat_features], axis=1)
    categorical_features = encoder.transform(categorical_features)
    processed_data = np.concatenate(
        [continuous_features, categorical_features], axis=1)

    # predicitons
    pred = model.predict(processed_data)
    prediction = label_binarizer.inverse_transform(pred)[0]

    return prediction
