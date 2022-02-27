# Script to train machine learning model.

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from starter.ml import data as data_utils
from starter.ml import model as model_utils
from starter.common import CATEGORICAL_FEATURES


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
    kf = KFold()
    fold_train_metrics = []
    fold_test_metrics = []
    for train_idx, test_idx in kf.split(data):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        # Process the test data with the process_data function.
        X_train, y_train, encoder, lb = data_utils.process_data(
            train,
            categorical_features=CATEGORICAL_FEATURES,
            label="salary",
            training=True,
        )

        X_test, y_test, _, _ = data_utils.process_data(
            test,
            categorical_features=CATEGORICAL_FEATURES,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Train and save a model.
        model = model_utils.train_model(X_train, y_train)
        preds_train = model_utils.inference(model, X_train)
        preds_test = model_utils.inference(model, X_test)
        metrics_train = model_utils.compute_model_metrics(y_train, preds_train)
        metrics_test = model_utils.compute_model_metrics(y_test, preds_test)
        fold_train_metrics.append(metrics_train)
        fold_test_metrics.append(metrics_test)

    metrics_train = np.mean(fold_train_metrics, axis=0)
    metrics_train_std = np.std(fold_train_metrics, axis=0)
    with open(metrics_dir / "train_metrics.json", "w") as fp:
        json.dump(
            {
                "avg_precision": metrics_train[0],
                "avg_recall": metrics_train[1],
                "avg_fbeta": metrics_train[2],
                "std_precision": metrics_train_std[0],
                "std_recall": metrics_train_std[1],
                "std_fbeta": metrics_train_std[2],
            },
            fp,
        )

    metrics_test = np.mean(fold_test_metrics, axis=0)
    metrics_test_std = np.std(fold_test_metrics, axis=0)
    with open(metrics_dir / "test_metrics.json", "w") as fp:
        json.dump(
            {
                "avg_precision": metrics_test[0],
                "avg_recall": metrics_test[1],
                "avg_fbeta": metrics_test[2],
                "std_precision": metrics_test_std[0],
                "std_recall": metrics_test_std[1],
                "std_fbeta": metrics_test_std[2],
            },
            fp,
        )

    pipeline = Pipeline([("encoder", encoder), ("random_forest", model)])
    model_utils.save(pipeline, model_dir / "model.pkl")
    model_utils.save(lb, model_dir / "label_encoder.pkl")


if __name__ == "__main__":
    args = get_args()
    train(args.infile, args.model_dir, args.metrics_dir)
