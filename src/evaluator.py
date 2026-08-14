import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    return {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def compare_models(models_dict: dict, X, y) -> pd.DataFrame:
    rows = []
    for name, model in models_dict.items():
        proba = model.predict_proba(X)
        m = compute_metrics(y, proba)
        m['model'] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index('model')
    return df[['roc_auc', 'pr_auc', 'precision', 'recall', 'f1']]


def plot_roc_curves(models_dict: dict, X, y) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models_dict.items():
        proba = model.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_pr_curves(models_dict: dict, X, y) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models_dict.items():
        proba = model.predict_proba(X)
        prec, rec, _ = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_proba, threshold: float = 0.5) -> plt.Figure:
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Legit', 'Fraud'],
        yticklabels=['Legit', 'Fraud'],
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (XGBoost)')
    plt.tight_layout()
    return fig
