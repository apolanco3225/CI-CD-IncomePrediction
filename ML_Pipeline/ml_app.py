"""
Script Machine Learning Pipeline
Author: Arturo Polanco
Date: February 2022
"""

import hydra
import logging
import os
from income_predictor.ml.data import clean_data
from income_predictor.train_model import split_data, train_save_model
from income_predictor.validate_model import validate_model
from omegaconf import DictConfig


_steps = [
    "data_cleaning",
    "train_model",
    "check_score"
]


@hydra.main(config_name="config.yml")
def go(config: DictConfig):
    """
    Run ml pipeline 
    """
    logging.basicConfig(level=logging.INFO)

    root_path = hydra.utils.get_original_cwd()
    artifacts_path = os.path.join(root_path, "model")
    
    # Steps to execute
    steps_par = config['main']['steps']
    data_path = config['data']['data_path']
    raw_data_path = os.path.join(root_path, data_path, "census.csv")
    clean_data_path = os.path.join(root_path, data_path, "clean_census.csv")


    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    cat_features = config['data']['cat_features']

    if "data_cleaning" in active_steps:
        logging.info("Cleaning data and saving clean version")
        clean_data(raw_data_path, clean_data_path)

    train_data, test_data = split_data(clean_data_path)

    if "train_model" in active_steps:
        logging.info("Train/Test model procedure started")
        train_save_model(train_data, artifacts_path, cat_features)

    if "check_score" in active_steps:
        logging.info("Score check procedure started")
        validate_model(test_data, cat_features, artifacts_path)


if __name__ == "__main__":
    """
    Main entrypoint
    """
    go()