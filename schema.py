"""
Script to set the data schema for API.
Author: Arturo Polanco
Date: February 2022
"""
   
from pydantic import BaseModel


class ModelInput(BaseModel):
    age: int
    workclass: str
    final_weight: int
    education: str
    user_education_id: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

