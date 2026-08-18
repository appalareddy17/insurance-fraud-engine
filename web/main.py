import io
import base64
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data, split_data
from src.feature_engineering import FeatureEngineer
from src.models.baseline import BaselineModel
from src.models.xgboost_model import XGBoostModel
from src import evaluator
from src.explainer import shap_explainer

DATA_PATH = Path(__file__).parent.parent / 'data' / 'insurance_claims.csv'
TEMPLATES_DIR = Path(__file__).parent / 'templates'
STATIC_DIR = Path(__file__).parent / 'static'

APP_STATE: dict = {}


def fig_to_b64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def risk_label(prob: float) -> tuple:
    """Return (label, tailwind_color) for a fraud probability."""
    if prob < 0.3:
        return "LOW", "green"
    elif prob < 0.6:
        return "MEDIUM", "amber"
    return "HIGH", "red"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place insurance_claims.csv in data/"
        )

    df = load_data(str(DATA_PATH))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    fe_baseline = FeatureEngineer()
    X_train_sc = fe_baseline.fit_transform_scaled(X_train)
    X_val_sc = fe_baseline.transform_scaled(X_val)

    fe_xgb = FeatureEngineer()
    X_train_enc = fe_xgb.fit_transform(X_train)
    X_val_enc = fe_xgb.transform(X_val)

    baseline = BaselineModel()
    baseline.train(X_train_sc, y_train)

    xgb = XGBoostModel(scale_pos_weight=3.0)
    xgb.train(X_train_enc, y_train)

    APP_STATE.update({
        "df": df,
        "fe_baseline": fe_baseline,
        "fe_xgb": fe_xgb,
        "X_train_enc": X_train_enc,
        "X_val_sc": X_val_sc,
        "X_val_enc": X_val_enc,
        "y_val": y_val,
        "baseline": baseline,
        "xgb": xgb,
    })
    yield
    APP_STATE.clear()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── GET / ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    df = APP_STATE["df"]
    fraud_count = int(df['fraud_reported'].sum())
    legit_count = len(df) - fraud_count

    fig_pie, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [legit_count, fraud_count],
        labels=['Legitimate', 'Fraud'],
        autopct='%1.1f%%',
        colors=['#4CAF50', '#F44336'],
        startangle=90,
    )
    pie_b64 = fig_to_b64(fig_pie)

    fig_hist, ax = plt.subplots(figsize=(6, 4))
    df[df['fraud_reported'] == 0]['total_claim_amount'].plot(
        kind='hist', bins=30, alpha=0.6, label='Legitimate', ax=ax, color='#4CAF50'
    )
    df[df['fraud_reported'] == 1]['total_claim_amount'].plot(
        kind='hist', bins=30, alpha=0.6, label='Fraud', ax=ax, color='#F44336'
    )
    ax.set_xlabel('Total Claim Amount (₹)')
    ax.set_ylabel('Count')
    ax.legend()
    hist_b64 = fig_to_b64(fig_hist)

    return templates.TemplateResponse("overview.html", {
        "request": request,
        "total": len(df),
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_pct": f"{fraud_count / len(df) * 100:.1f}",
        "legit_pct": f"{legit_count / len(df) * 100:.1f}",
        "pie_b64": pie_b64,
        "hist_b64": hist_b64,
        "sample": df.head(10).to_dict(orient='records'),
        "columns": df.columns.tolist(),
    })


# ── GET /evaluation ────────────────────────────────────────────────────────────
@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation(request: Request):
    baseline = APP_STATE["baseline"]
    xgb = APP_STATE["xgb"]
    X_val_sc = APP_STATE["X_val_sc"]
    X_val_enc = APP_STATE["X_val_enc"]
    y_val = APP_STATE["y_val"]

    class _Proxy:
        def __init__(self, model, X):
            self._model = model
            self._X = X

        def predict_proba(self, _):
            return self._model.predict_proba(self._X)

    models_proxy = {
        'Logistic Regression': _Proxy(baseline, X_val_sc),
        'XGBoost': _Proxy(xgb, X_val_enc),
    }

    metrics_df = evaluator.compare_models(models_proxy, None, y_val)

    # Pre-format numbers for template
    metrics_records = []
    for model_name, row in metrics_df.iterrows():
        record = {"model": model_name}
        for col in metrics_df.columns:
            record[col] = f"{row[col]:.3f}"
        metrics_records.append(record)

    fig_roc = evaluator.plot_roc_curves(models_proxy, None, y_val)
    fig_pr = evaluator.plot_pr_curves(models_proxy, None, y_val)
    fig_cm = evaluator.plot_confusion_matrix(y_val, xgb.predict_proba(X_val_enc))

    return templates.TemplateResponse("evaluation.html", {
        "request": request,
        "metrics": metrics_records,
        "columns": metrics_df.columns.tolist(),
        "roc_b64": fig_to_b64(fig_roc),
        "pr_b64": fig_to_b64(fig_pr),
        "cm_b64": fig_to_b64(fig_cm),
    })


# ── GET /explainability ────────────────────────────────────────────────────────
@app.get("/explainability", response_class=HTMLResponse)
async def explainability(request: Request, idx: int = 0):
    xgb = APP_STATE["xgb"]
    X_train_enc = APP_STATE["X_train_enc"]
    X_val_enc = APP_STATE["X_val_enc"]
    y_val = APP_STATE["y_val"]

    idx = max(0, min(idx, len(X_val_enc) - 1))

    sample_size = min(500, len(X_train_enc))
    X_sample = X_train_enc.sample(sample_size, random_state=42)
    fig_bee = shap_explainer.plot_beeswarm(xgb, X_sample, max_display=15)
    bee_b64 = fig_to_b64(fig_bee)

    X_single = X_val_enc.iloc[[idx]]
    fig_wf = shap_explainer.plot_waterfall(xgb, X_single)
    wf_b64 = fig_to_b64(fig_wf)

    prob = float(xgb.predict_proba(X_single)[0])
    actual = "Fraud" if int(y_val.iloc[idx]) == 1 else "Legitimate"
    label, color = risk_label(prob)

    return templates.TemplateResponse("explainability.html", {
        "request": request,
        "bee_b64": bee_b64,
        "wf_b64": wf_b64,
        "idx": idx,
        "max_idx": len(X_val_enc) - 1,
        "prob": f"{prob:.3f}",
        "actual": actual,
        "risk_label": label,
        "risk_color": color,
    })


# ── GET /scorer ────────────────────────────────────────────────────────────────
@app.get("/scorer", response_class=HTMLResponse)
async def scorer_get(request: Request):
    return templates.TemplateResponse("scorer.html", {
        "request": request,
        "result": None,
        "error": None,
    })


# ── POST /scorer ───────────────────────────────────────────────────────────────
@app.post("/scorer", response_class=HTMLResponse)
async def scorer_post(
    request: Request,
    months_as_customer: int = Form(...),
    age: int = Form(...),
    policy_deductable: int = Form(...),
    policy_annual_premium: float = Form(...),
    umbrella_limit: int = Form(...),
    insured_sex: str = Form(...),
    insured_education_level: str = Form(...),
    insured_occupation: str = Form(...),
    insured_relationship: str = Form(...),
    incident_type: str = Form(...),
    collision_type: str = Form(...),
    incident_severity: str = Form(...),
    incident_hour_of_the_day: int = Form(...),
    number_of_vehicles_involved: int = Form(...),
    bodily_injuries: int = Form(...),
    witnesses: int = Form(...),
    police_report_available: str = Form(...),
    property_damage: str = Form(...),
    total_claim_amount: int = Form(...),
    injury_claim: int = Form(...),
    property_claim: int = Form(...),
    vehicle_claim: int = Form(...),
    auto_make: str = Form(...),
    auto_year: int = Form(...),
):
    fe_xgb = APP_STATE["fe_xgb"]
    xgb = APP_STATE["xgb"]

    input_data = pd.DataFrame([{
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_deductable': policy_deductable,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'insured_sex': insured_sex,
        'insured_education_level': insured_education_level,
        'insured_occupation': insured_occupation,
        'insured_relationship': insured_relationship,
        'incident_type': incident_type,
        'collision_type': None if collision_type == 'NA' else collision_type,
        'incident_severity': incident_severity,
        'incident_hour_of_the_day': incident_hour_of_the_day,
        'number_of_vehicles_involved': number_of_vehicles_involved,
        'bodily_injuries': bodily_injuries,
        'witnesses': witnesses,
        'police_report_available': police_report_available,
        'property_damage': property_damage,
        'total_claim_amount': total_claim_amount,
        'injury_claim': injury_claim,
        'property_claim': property_claim,
        'vehicle_claim': vehicle_claim,
        'auto_make': auto_make,
        'auto_year': auto_year,
    }])

    X_input = fe_xgb.transform(input_data)
    prob = float(xgb.predict_proba(X_input)[0])
    label, color = risk_label(prob)

    fig_wf = shap_explainer.plot_waterfall(xgb, X_input)
    wf_b64 = fig_to_b64(fig_wf)

    result = {
        "probability": f"{prob:.1%}",
        "prob_pct": f"{prob * 100:.1f}%",
        "risk_label": label,
        "risk_color": color,
        "waterfall_b64": wf_b64,
    }

    return templates.TemplateResponse("scorer.html", {
        "request": request,
        "result": result,
        "error": None,
    })
