import pandas as pd
import numpy as np
import pytest
from src.models.baseline import BaselineModel
from src.models.xgboost_model import XGBoostModel


@pytest.fixture
def small_dataset():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 50))
    return X, y


def test_baseline_trains_without_error(small_dataset):
    X, y = small_dataset
    model = BaselineModel()
    model.train(X, y)
    assert model.model is not None


def test_baseline_predict_proba_range(small_dataset):
    X, y = small_dataset
    model = BaselineModel()
    model.train(X, y)
    proba = model.predict_proba(X)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert len(proba) == len(X)


def test_baseline_predict_binary(small_dataset):
    X, y = small_dataset
    model = BaselineModel()
    model.train(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


def test_xgboost_trains_without_error(small_dataset):
    X, y = small_dataset
    model = XGBoostModel(scale_pos_weight=3.0)
    model.train(X, y)
    assert model.model is not None


def test_xgboost_predict_proba_range(small_dataset):
    X, y = small_dataset
    model = XGBoostModel(scale_pos_weight=3.0)
    model.train(X, y)
    proba = model.predict_proba(X)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert len(proba) == len(X)


def test_xgboost_predict_binary(small_dataset):
    X, y = small_dataset
    model = XGBoostModel(scale_pos_weight=3.0)
    model.train(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})
