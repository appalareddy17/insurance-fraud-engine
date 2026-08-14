import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd


def plot_beeswarm(xgb_model, X: pd.DataFrame, max_display: int = 15) -> plt.Figure:
    """Global feature importance via SHAP beeswarm. X should be the encoded training set."""
    explainer = shap.TreeExplainer(xgb_model.model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_waterfall(xgb_model, X_row: pd.DataFrame) -> plt.Figure:
    """Local explanation for a single claim row via SHAP waterfall."""
    explainer = shap.TreeExplainer(xgb_model.model)
    shap_explanation = explainer(X_row)
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_explanation[0], show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig
