import pytest
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

from starter.ml import data as data_utils


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": ["1", "2", "3", "4"],
            "C": ["yes", "no", "yes", "yes"],
        }
    )


def test__catlist_label_training(df: pd.DataFrame):
    x, y, encoder, lb = data_utils.process_data(
        X=df, categorical_features=["B"], label="C", training=True
    )

    assert np.issubdtype(y.dtype, np.signedinteger)
    assert np.issubdtype(x.dtype, np.number)
    check_is_fitted(encoder)
    check_is_fitted(lb)


def test__catlist_nolabel_training(df: pd.DataFrame):
    with pytest.raises(AttributeError) as ex_info:
        x, y, encoder, lb = data_utils.process_data(
            X=df, categorical_features=["B"], label=None, training=True
        )

    assert "object has no attribute 'values'" in str(ex_info.value)


def test__catlist_label_notraining(df: pd.DataFrame):
    with pytest.raises(AttributeError) as ex_info:
        x, y, encoder, lb = data_utils.process_data(
            X=df, categorical_features=["B"], label="C", training=False
        )

    assert "object has no attribute 'transform'" in str(ex_info.value)


def test__catlist_nolabel_notraining(df: pd.DataFrame):
    x, y, encoder, lb = data_utils.process_data(
        X=df, categorical_features=["B"], label="C", training=True
    )

    _x, _y, _encoder, _lb = data_utils.process_data(
        X=df.drop(["C"], axis=1),
        categorical_features=["B"],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert _y.shape == (0,)
    assert x.shape == _x.shape
    assert np.allclose(x, _x)
