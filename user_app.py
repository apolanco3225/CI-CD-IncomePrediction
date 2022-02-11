"""
Script to serve income prediction model as API.
Author: Arturo Polanco
Date: February 2022
"""
   
import os
import yaml
from joblib import load
from schema import ModelInput
import pandas as pd
import numpy as np
from fastapi import FastAPI

from ml_pipeline.income_predictor.ml.data import process_data
from ml_pipeline.income_predictor.ml.model import inference
from ml_pipeline.income_predictor.inference import run_inference



if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")



app = FastAPI()



@app.get("/")
async def get_items():
    return {"message": "Aloha!"}


@app.post("/predict")
async def inference(sample:ModelInput):



    prediction = inference(sample)

    #prediction = "holi"
    return {"prediction": prediction}


def inference(sample):




    with open('ml_pipeline/config.yml') as f:
        config = yaml.load(f)

    cat_features = config["data"]["cat_features"]
    artifacts_path = "ml_pipeline/model"



    # define reading paths
    model_path = os.path.join(artifacts_path, "model.joblib")
    encoder_path = os.path.join(artifacts_path, "encoder.joblib")
    label_binarizer_path = os.path.join(artifacts_path, "label_binarizer.joblib")

    model = load(model_path)
    encoder = load(encoder_path)
    label_binarizer = load(label_binarizer_path)
    cat_features = config['data']['cat_features']


    sample = sample.dict()

    data = pd.DataFrame(data=sample.values(), index=sample.keys()).T

    # data preprocessing
    categorical_features = data[cat_features].values
    continuous_features = data.drop(*[cat_features], axis=1)
    categorical_features = encoder.transform(categorical_features)
    processed_data = np.concatenate([continuous_features, categorical_features], axis=1)



    pred = model.predict(processed_data)
    prediction = label_binarizer.inverse_transform(pred)[0]
    return prediction
