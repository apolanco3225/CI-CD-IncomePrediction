import pytest
import yaml
import pandas as pd
import yaml

from ml_pipeline.income_predictor.ml.data import data_cleaning_steps



with open('ml_pipeline/config.yml', encoding='utf8') as file:
    config = yaml.safe_load(file)

@pytest.fixture
def raw_data():
    """
    Read raw data
    """
    raw_dataset = pd.read_csv("ml_pipeline/data/census.csv")

    return raw_dataset

@pytest.fixture
def column_names():
    return config["infer"]["columns"]

@pytest.fixture
def cat_features_values_dict():
    return cat_features_values




@pytest.fixture
def clean_data(raw_data):
    """
    Get dataset
    """
    clean_dataframe = data_cleaning_steps(raw_data)
    return clean_dataframe


@pytest.fixture
def cat_features():
    """
    Get dataset
    """
    with open('config.yml') as f:
        config = yaml.load(f)

    return config['data']['cat_features']


cat_features_values = {'workclass': ['State-gov',
  'Self-emp-not-inc',
  'Private',
  'Federal-gov',
  'Local-gov',
  'Self-emp-inc',
  'Without-pay'],
 'education': ['Bachelors',
  'HS-grad',
  '11th',
  'Masters',
  '9th',
  'Some-college',
  'Assoc-acdm',
  '7th-8th',
  'Doctorate',
  'Assoc-voc',
  'Prof-school',
  '5th-6th',
  '10th',
  'Preschool',
  '12th',
  '1st-4th'],
 'marital_status': ['Never-married',
  'Married-civ-spouse',
  'Divorced',
  'Married-spouse-absent',
  'Separated',
  'Married-AF-spouse',
  'Widowed'],
 'occupation': ['Adm-clerical',
  'Exec-managerial',
  'Handlers-cleaners',
  'Prof-specialty',
  'Other-service',
  'Sales',
  'Transport-moving',
  'Farming-fishing',
  'Machine-op-inspct',
  'Tech-support',
  'Craft-repair',
  'Protective-serv',
  'Armed-Forces',
  'Priv-house-serv'],
 'relationship': ['Not-in-family',
  'Husband',
  'Wife',
  'Own-child',
  'Unmarried',
  'Other-relative'],
 'race': ['White',
  'Black',
  'Asian-Pac-Islander',
  'Amer-Indian-Eskimo',
  'Other'],
 'sex': ['Male', 'Female'],
 'native_country': ['United-States',
  'Cuba',
  'Jamaica',
  'India',
  'Mexico',
  'Puerto-Rico',
  'Honduras',
  'England',
  'Canada',
  'Germany',
  'Iran',
  'Philippines',
  'Poland',
  'Columbia',
  'Cambodia',
  'Thailand',
  'Ecuador',
  'Laos',
  'Taiwan',
  'Haiti',
  'Portugal',
  'Dominican-Republic',
  'El-Salvador',
  'France',
  'Guatemala',
  'Italy',
  'China',
  'South',
  'Japan',
  'Yugoslavia',
  'Peru',
  'Outlying-US(Guam-USVI-etc)',
  'Scotland',
  'Trinadad&Tobago',
  'Greece',
  'Nicaragua',
  'Vietnam',
  'Hong',
  'Ireland',
  'Hungary',
  'Holand-Netherlands']
  }