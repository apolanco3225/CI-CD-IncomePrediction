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

    class Config:
        schema_extra = {
            "example": {
                "age":42,
                "workclass":"Private",
                "final_weight":159449,
                "education":"Bachelors",
                "user_education_id":13,
                "marital_status":"Married-civ-spouse",
                "occupation":"Exec-managerial",
                "relationship":"Husband",
                "race":"White",
                "sex":"Male",
                "capital_gain":5178,
                "capital_loss":0,
                "hours_per_week":40,
                "native_country":"United-States"
            }
        }

