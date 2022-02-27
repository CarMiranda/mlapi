from typing import Any
import numpy as np
from numpy import typing as npt
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer

from starter.common import CATEGORICAL_FEATURES



def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder: ColumnTransformer = None,
    lb=None,
):
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.compose.ColumnTransformer
        Trained sklearn ColumnTransformer, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.compose.ColumnTransformer
        Trained ColumnTransformer if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    if training is True:
        encoder = ColumnTransformer(
            [
                (
                    "categories",
                    OneHotEncoder(dtype="int", sparse=False, handle_unknown="ignore"),
                    categorical_features,
                )
            ],
            remainder="passthrough",
        )
        lb = LabelBinarizer()
        X = encoder.fit_transform(X)
        y = lb.fit_transform(y.values).ravel()
    else:
        X = encoder.transform(X)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    return X, y, encoder, lb
