# Script to train machine learning model.

import argparse
import json
import pathlib
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from starter.ml import data as data_utils
from starter.ml import model as model_utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--infile", type=pathlib.Path, help="Path to input file (clean data)."
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=pathlib.Path,
        help="Directory to output model artifacts.",
        default="model",
    )
    parser.add_argument(
        "-M",
        "--metrics-dir",
        type=pathlib.Path,
        help="Directory to output metrics artifacts.",
        default="metrics",
    )

    return parser.parse_args()


def train(data_path: pathlib.Path, model_dir: pathlib.Path, metrics_dir: pathlib.Path):
    """_summary_

    Args:
        data_path (pathlib.Path): _description_
        model_dir (pathlib.Path): _description_
        metrics_dir (pathlib.Path): _description_
    """

    # Create output directory
    model_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    # Add code to load in the data.
    data = pd.read_csv(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
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

    # Process the test data with the process_data function.
    X_train, y_train, encoder, lb = data_utils.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = data_utils.process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train and save a model.
    model = model_utils.train_model(X_train, y_train)
    preds_train = model_utils.inference(model, X_train)
    metrics_train = model_utils.compute_model_metrics(y_train, preds_train)
    with open(metrics_dir / "train_metrics.json", "w") as fp:
        json.dump(
            {
                "precision": metrics_train[0],
                "recall": metrics_train[1],
                "fbeta": metrics_train[2],
            },
            fp,
        )

    preds_test = model_utils.inference(model, X_test)
    metrics_test = model_utils.compute_model_metrics(y_test, preds_test)
    with open(metrics_dir / "test_metrics.json", "w") as fp:
        json.dump(
            {
                "precision": metrics_test[0],
                "recall": metrics_test[1],
                "fbeta": metrics_test[2],
            },
            fp,
        )

    pipeline = Pipeline([("encoder", encoder), ("random_forest", model)])
    model_utils.save(pipeline, model_dir / "model.pkl")


if __name__ == "__main__":
    args = get_args()
    train(args.infile, args.model_dir, args.metrics_dir)
