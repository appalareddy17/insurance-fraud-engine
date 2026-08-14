import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import load_data, split_data
from src.feature_engineering import FeatureEngineer
from src.models.baseline import BaselineModel
from src.models.xgboost_model import XGBoostModel
from src import evaluator

DATA_PATH = 'data/insurance_claims.csv'


@pytest.fixture(scope='module')
def trained_models():
    df = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    fe_b = FeatureEngineer()
    X_train_sc = fe_b.fit_transform_scaled(X_train)
    X_val_sc = fe_b.transform_scaled(X_val)

    fe_x = FeatureEngineer()
    X_train_enc = fe_x.fit_transform(X_train)
    X_val_enc = fe_x.transform(X_val)

    baseline = BaselineModel()
    baseline.train(X_train_sc, y_train)

    xgb = XGBoostModel(scale_pos_weight=3.0)
    xgb.train(X_train_enc, y_train)

    return baseline, xgb, X_val_sc, X_val_enc, y_val


def test_baseline_roc_auc_above_random(trained_models):
    baseline, _, X_val_sc, _, y_val = trained_models
    proba = baseline.predict_proba(X_val_sc)
    metrics = evaluator.compute_metrics(y_val, proba)
    assert metrics['roc_auc'] > 0.5


def test_xgboost_roc_auc_above_random(trained_models):
    _, xgb, _, X_val_enc, y_val = trained_models
    proba = xgb.predict_proba(X_val_enc)
    metrics = evaluator.compute_metrics(y_val, proba)
    assert metrics['roc_auc'] > 0.5


def test_compare_models_returns_dataframe(trained_models):
    baseline, xgb, X_val_sc, X_val_enc, y_val = trained_models

    class _Proxy:
        def __init__(self, model, X):
            self._model = model
            self._X = X

        def predict_proba(self, _):
            return self._model.predict_proba(self._X)

    models = {
        'Logistic Regression': _Proxy(baseline, X_val_sc),
        'XGBoost': _Proxy(xgb, X_val_enc),
    }
    df = evaluator.compare_models(models, None, y_val)
    assert 'roc_auc' in df.columns
    assert len(df) == 2


def test_plot_roc_returns_figure(trained_models):
    baseline, xgb, X_val_sc, X_val_enc, y_val = trained_models

    class _Proxy:
        def __init__(self, model, X):
            self._model = model
            self._X = X

        def predict_proba(self, _):
            return self._model.predict_proba(self._X)

    models = {
        'Logistic Regression': _Proxy(baseline, X_val_sc),
        'XGBoost': _Proxy(xgb, X_val_enc),
    }
    fig = evaluator.plot_roc_curves(models, None, y_val)
    assert isinstance(fig, plt.Figure)
    plt.close('all')
