import pathlib
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

from starter.ml import data as data_utils
from starter.ml import model as model_utils
from starter import train_model


@pytest.fixture
def training_preprocess():
    df = pd.DataFrame(
        {
            "A": list(range(10)),
            "B": list(map(str, range(10))),
            "C": ["yes" if i % 2 == 0 else "no" for i in range(10)],
        }
    )

    return data_utils.process_data(df, ["B"], "C", training=True)


@pytest.fixture
def raw_data():
    return pd.DataFrame(
        {
            "A": list(range(10)),
            "B": list(map(str, range(10))),
            "C": ["yes" if i % 2 == 0 else "no" for i in range(10)],
        }
    )


def test__slice(raw_data: pd.DataFrame):
    indices = data_utils.get_slice_indices(raw_data, "C", "yes")

    assert indices.sum() == 5
    assert raw_data[indices].A.sum() == 20


def test__train(training_preprocess):
    x, y, encoder, lb = training_preprocess

    model = model_utils.train_model(x, y)
    check_is_fitted(model)

    preds = model_utils.inference(model, x)
    assert preds.shape[0] == x.shape[0]
    assert np.issubdtype(preds.dtype, np.signedinteger)

    metrics = model_utils.compute_model_metrics(y, preds)
    assert len(metrics) == 3
    assert all(0 <= m <= 1 for m in metrics)


def test__full_train():

    data = {
        "workclass": [""] * 10,
        "education": [""] * 10,
        "marital-status": [""] * 10,
        "occupation": [""] * 10,
        "relationship": [""] * 10,
        "race": [""] * 10,
        "sex": [""] * 10,
        "native-country": [""] * 10,
        "salary": ["<=1"] * 5 + [">1"] * 5,
    }
    df = pd.DataFrame(data)
    data_path = pathlib.Path(tempfile.mkstemp()[1])
    model_dir = pathlib.Path(tempfile.mkdtemp())
    metrics_dir = pathlib.Path(tempfile.mkdtemp())
    df.to_csv(data_path)

    train_model.train(data_path, model_dir, metrics_dir)

    assert len(list(metrics_dir.iterdir())) == 2
    assert len(list(model_dir.iterdir())) == 2

    data_path.unlink()
    shutil.rmtree(metrics_dir)
    shutil.rmtree(model_dir)


def test__save_load(training_preprocess):
    x, y, encoder, lb = training_preprocess
    model_path = pathlib.Path(tempfile.mkstemp()[1])

    model = model_utils.train_model(x, y)
    preds = model_utils.inference(model, x)
    model_utils.save(model, model_path)

    _model = model_utils.load(model_path)
    _preds = model_utils.inference(_model, x)

    assert np.allclose(preds, _preds)
