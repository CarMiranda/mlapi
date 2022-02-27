import argparse
import json
import pathlib
from typing import Any, List
import pandas as pd
import numpy as np
from numpy import typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from starter.ml import data as data_utils
from starter.ml import model as model_utils
from starter.common import CATEGORICAL_FEATURES


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--infile",
        type=pathlib.Path,
        help="Path to input file (clean data).",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=pathlib.Path,
        help="Directory containing a model and a label encoder artifacts.",
        default="model",
        required=True,
    )
    parser.add_argument(
        "-M",
        "--metrics-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to output metrics artifacts.",
        required=False,
    )
    parser.add_argument(
        "-k",
        "--slice-key",
        type=str,
        default=None,
        help="Name of a column to take a slice from.",
        required=False,
    )
    parser.add_argument(
        "-v",
        "--slice-value",
        type=str,
        default=None,
        help="Value of the column to filter the slice.",
        required=False,
    )

    return parser.parse_args()


def evaluate_classifier(
    model: RandomForestClassifier,
    data: pd.DataFrame,
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
    slice: npt.NDArray[np.bool_] = None,
):

    if slice is not None:
        data = data[slice].copy()

    x, y, encoder, lb = data_utils.process_data(
        data,
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = model_utils.inference(model, x)
    metrics = model_utils.compute_model_metrics(y, preds)

    return metrics


def evaluate_pipeline(
    pipeline: Pipeline,
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.unsignedinteger],
    slice: npt.NDArray[np.bool_] = None,
):
    if slice is not None:
        x = x[slice].copy()
        y = y[slice].copy()

    preds = model_utils.inference(pipeline, x)
    metrics = model_utils.compute_model_metrics(y, preds)

    return metrics


def evaluate(
    data_path: pathlib.Path,
    model_dir: pathlib.Path,
    metrics_dir: pathlib.Path = None,
    slice_key: str = None,
    slice_value: Any = None,
):
    df = pd.read_csv(data_path)
    model = model_utils.load(model_dir / "model.pkl")
    lb = model_utils.load(model_dir / "label_encoder.pkl")

    data_slice = None
    if slice_key is not None:
        if slice_value is not None:
            slice_values = [slice_value]
        else:
            slice_values = df[slice_key].unique()

    if data_slice is None and slice_key is None:
        metrics = evaluate_pipeline(
            model, df.drop(["salary"], axis=1), lb.transform(df.salary)
        )
        if metrics_dir is not None:
            with open(
                metrics_dir / f"{slice_key}_{slice_value}_metrics.json", "w"
            ) as fp:
                json.dump(
                    {
                        "precision": metrics[0],
                        "recall": metrics[1],
                        "fbeta": metrics[2],
                    },
                    fp,
                )
        metrics = {"Overall": metrics}
    else:
        metrics = dict()
        for slice_value in slice_values:
            data_slice = data_utils.get_slice_indices(df, slice_key, slice_value)

            slice_metrics = evaluate_pipeline(
                model, df.drop(["salary"], axis=1), lb.transform(df.salary), data_slice
            )
            if metrics_dir is not None:
                with open(
                    metrics_dir / f"{slice_key}_{slice_value}_metrics.json", "w"
                ) as fp:
                    json.dump(
                        {
                            "precision": slice_metrics[0],
                            "recall": slice_metrics[1],
                            "fbeta": slice_metrics[2],
                        },
                        fp,
                    )
            metrics[slice_value] = slice_metrics

    return metrics


if __name__ == "__main__":
    args = get_args()
    metrics = evaluate(
        args.infile, args.model_dir, args.metrics_dir, args.slice_key, args.slice_value
    )

    print(",".join(["Value", "Precision", "Recall", "FBeta"]))
    for name, (precision, recall, fbeta) in metrics.items():
        print(",".join([name, f"{precision:.5f}", f"{recall:.5f}", f"{fbeta:.5f}"]))
