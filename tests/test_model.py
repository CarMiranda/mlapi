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
            "A": [1, 2, 3, 4],
            "B": ["1", "2", "3", "4"],
            "C": ["yes", "no", "yes", "yes"],
        }
    )

    return data_utils.process_data(df, ["B"], "C", training=True)


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
        "workclass": [""] * 4,
        "education": [""] * 4,
        "marital-status": [""] * 4,
        "occupation": [""] * 4,
        "relationship": [""] * 4,
        "race": [""] * 4,
        "sex": [""] * 4,
        "native-country": [""] * 4,
        "salary": ["<=1"] * 2 + [">1"] * 2,
    }
    df = pd.DataFrame(data)
    data_path = pathlib.Path(tempfile.mkstemp()[1])
    model_dir = pathlib.Path(tempfile.mkdtemp())
    metrics_dir = pathlib.Path(tempfile.mkdtemp())
    df.to_csv(data_path)

    train_model.train(data_path, model_dir, metrics_dir)

    assert len(list(metrics_dir.iterdir())) == 2
    assert len(list(model_dir.iterdir())) == 1

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
