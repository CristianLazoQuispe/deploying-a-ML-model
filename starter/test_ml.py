import os
import pytest
import pandas as pd

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model,inference,compute_model_metrics
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def data():
    """
    Read dataset
    Returns:
        data: pd.DataFrame, census data cleaned
    """
    df = pd.read_csv("starter/data/census_cleaned.csv")
    return df


def test_data_size(data):
    """
    Testing the size of columns and rows
    """
    assert data.shape[0]>0
    assert data.shape[1] == 15

def test_models_files():
    
    assert os.path.isfile("starter/model/model.pkl")
    assert os.path.isfile("starter/model/encoder.pkl")
    assert os.path.isfile("starter/model/lb.pkl")
    
def test_training(data):
    train, test = train_test_split(data, test_size=0.20)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # Train and save a model encoder and lb Binarizer

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)
    
    assert len(metrics) == 3
    for metric in metrics:
        assert metric >=0.8
