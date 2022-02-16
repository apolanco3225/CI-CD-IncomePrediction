
"""
This module:
1. Creates a model for income prediction
2. Computes metrics of the model in the whole dataset
3. Computes metrics of the model in slices of the dataset
4. Make predictions
Author: Arturo Polanco
Date: February 2022
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from .data import process_data
import logging 
import os
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    model = LogisticRegression(solver='lbfgs', max_iter=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logging.info(f"Accuracy mean: {np.mean(scores)}")
    logging.info(f"Accuracy std: {np.std(scores)}")

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def scores(
    model, 
    test_data, 
    encoder,
    lb, 
    cat_features, 
    ):
    """
    Compute score in test set
    ------
    model: joblib file
        Trained model that will be tested
    test_features: np.array
        Features in the test set
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Return
        precision_score
        recall_score
        f_beta_score
        
    -------
    """

    test_features, test_label, _, _ = process_data(
        test_data,
        categorical_features=cat_features, training=False,
        label="salary", encoder=encoder, lb=lb)

    predictions = model.predict(test_features)

    precision_score, recall_score, f_beta_score = compute_model_metrics(test_label, predictions)

    return precision_score, recall_score, f_beta_score




def scores_in_slices(
    model, 
    test_data, 
    encoder,
    lb, 
    cat_features, 
    output_path):
    """
    Compute score in categorical features
    Inputs
    ------
    model: joblib file
        Trained model that will be tested
    test_features: np.array
        Features in the test set
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Return
        None
        Text is written in txt format
    -------
    """
    logs_path = os.path.join(output_path,'slice_logs.txt')
    with open(logs_path, 'w') as file:
        for category in cat_features:
            for cls in test_data[category].unique():
                slice_df = test_data[test_data[category] == cls]

                test_features, test_label, _, _ = process_data(
                    slice_df,
                    categorical_features=cat_features, training=False,
                    label="salary", encoder=encoder, lb=lb)

                predictions = model.predict(test_features)

                precision_score, recall_score, f_beta_score = compute_model_metrics(test_label, predictions)

                metrics_slice = f"{category} - Precision:{precision_score} Recall:{recall_score} F Beta Score: {f_beta_score}"

                #logging.info(metrics_slice)
                file.write(metrics_slice  + '\n')