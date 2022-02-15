"""
Script to serve income prediction model as API.
Author: Arturo Polanco
Date: February 2022
"""
# import necessary packages
import os
import sys
from fastapi import FastAPI
from schema import ModelInput
from inference_pipeline import inference




# instantiating fastapi
app = FastAPI()


@app.get("/")
async def get_items():
    """
    Greetings endpoint
    """
    return {"message": "Aloha!"}


@app.post("/predict/")
async def model_inference(sample: ModelInput):
    """
    Prediction endpoint
    """
    prediction = inference(sample)

    return {"prediction": prediction}
    