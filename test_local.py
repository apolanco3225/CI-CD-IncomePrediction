"""
Script to test API.
Author: Arturo Polanco
Date: February 2022
"""

from fastapi.testclient import TestClient

# Import our app from main.py.
from user_app import app
# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    request = client.get("/")
    assert request.status_code == 200

def test_post_lower():

    request = client.post("/predict/", json={
        "age":39,
        "workclass":"State-gov",
        "final_weight":77516,
        "education":"Bachelors",
        "user_education_id":13,
        "marital_status":"Never-married",
        "occupation":"Adm-clerical",
        "relationship":"Not-in-family",
        "race":"White",
        "sex":"Male",
        "capital_gain":2174,
        "capital_loss":0,
        "hours_per_week":40,
        "native_country":"United-States"}
)
    assert request.status_code == 200
    assert request.json() == {"prediction": "<=50K"}

def test_post_upper():

    request = client.post("/predict/", json={
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
        })
    assert request.status_code == 200
    assert request.json() == {"prediction": ">50K"}