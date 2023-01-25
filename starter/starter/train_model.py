# Script to train machine learning model.

from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

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


def get_data():
    # Add code to load in the data.
    data = pd.read_csv("starter/data/census_cleaned.csv")
    return data


def split_data(data):
    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20,random_state=42)
    return train, test


def pprocessing_data(train):
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train, encoder, lb


def train_all(X_train, y_train, encoder, lb):
    # Train and save a model encoder and lb Binarizer

    model = train_model(X_train, y_train)
    pd.to_pickle(model, "starter/model/model.pkl")
    pd.to_pickle(encoder, "starter/model/encoder.pkl")
    pd.to_pickle(lb, "starter/model/lb.pkl")

    return model


def inference_test(model, X_train, y_train):
    # Make an inference
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)
    return metrics


if __name__ == "__main__":

    data = get_data()
    train, test = train_test_split(data, test_size=0.20,random_state=42)
    X_train, y_train, encoder, lb = pprocessing_data(train)
    X_test, y_test, encoder, lb   = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)


    model = train_all(X_train, y_train, encoder, lb)
    metrics = inference_test(model, X_train, y_train)
    print("Training metrics: precision={:1.4f} recall={:1.4f} fbeta={:1.4f}".format(metrics[0],metrics[1],metrics[2]))
    metrics = inference_test(model, X_test, y_test)
    print("Testing  metrics: precision={:1.4f} recall={:1.4f} fbeta={:1.4f}".format(metrics[0],metrics[1],metrics[2]))
