# Insurance Fraud Claims Detection Engine

A machine learning prototype that detects fraudulent auto insurance claims using **Logistic Regression** and **XGBoost**, with full **SHAP explainability** and a live web interface built on **FastAPI + Tailwind CSS**.

---

## Live Demo

**https://insurance-fraud-engine.onrender.com**

> First load may take ~30 seconds (cold start — models train on startup).

---

## Pages

| Page | Route | Description |
|---|---|---|
| Overview | `/` | Dataset stats, fraud distribution charts, sample table |
| Evaluation | `/evaluation` | ROC curves, PR curves, confusion matrix, metrics table |
| Explainability | `/explainability` | SHAP beeswarm (global) + waterfall (per-claim) |
| Claim Scorer | `/scorer` | Enter claim details → get fraud probability + SHAP reason |

---

## Project Structure

```
insurance-fraud-engine/
├── data/
│   └── insurance_claims.csv              # 1,000 rows, 39 features
├── src/
│   ├── data_loader.py                    # Load, clean, stratified split
│   ├── feature_engineering.py            # Encode, scale, derived features
│   ├── evaluator.py                      # Metrics + matplotlib charts
│   ├── models/
│   │   ├── baseline.py                   # Logistic Regression
│   │   └── xgboost_model.py              # XGBoost + GridSearchCV
│   └── explainer/
│       └── shap_explainer.py             # SHAP beeswarm + waterfall
├── web/
│   ├── main.py                           # FastAPI — 4 routes
│   └── templates/                        # Jinja2 + Tailwind CSS
│       ├── base.html
│       ├── overview.html
│       ├── evaluation.html
│       ├── explainability.html
│       └── scorer.html
├── tests/                                # 28 pytest tests (21 ML + 7 web)
├── Insurance_Fraud_Detection_Report.ipynb  # Full project report notebook
├── Insurance_Fraud_Detection_Engine.zip    # Complete project zip
├── Dockerfile                            # Docker build for Render
├── render.yaml                           # Render deployment config
├── requirements.txt                      # ML dependencies
└── requirements_web.txt                  # Web dependencies
```

---

## Models

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | ~0.78 | ~0.73 | ~0.55 | ~0.63 |
| XGBoost (tuned) | ~0.87 | ~0.76 | ~0.63 | ~0.69 |

- **Class imbalance** handled via `class_weight='balanced'` (LR) and `scale_pos_weight=3.0` (XGBoost)
- **GridSearchCV** tuned: `n_estimators`, `max_depth`, `learning_rate`
- **Risk thresholds:** LOW < 0.30 | MEDIUM 0.30–0.60 | HIGH ≥ 0.60

---

## Key Features

- **3 Derived Features:** `claim_to_premium_ratio`, `is_night_incident`, `no_police_no_witness`
- **SHAP Global View:** Beeswarm plot showing top 15 features driving predictions
- **SHAP Local View:** Waterfall chart explaining each individual claim
- **Two FeatureEngineer instances:** scaled for Logistic Regression, unscaled for XGBoost
- **Stratified split:** 70% train / 15% val / 15% test

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt -r requirements_web.txt

# Start the web app
uvicorn web.main:app --reload --port 8000
```

Open `http://localhost:8000`

---

## Run Tests

```bash
pip install pytest
pytest tests/ -v
```

28 tests — 21 ML pipeline tests + 7 web utility tests.

---

## Dataset

**Auto Insurance Claims** — 1,000 records, 39 columns
Source: [Kaggle — buntyshah/auto-insurance-claims-data](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)
Included in `data/insurance_claims.csv`

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn, XGBoost, SHAP |
| Data | pandas, numpy |
| Web | FastAPI, Jinja2, Tailwind CSS |
| Charts | matplotlib, seaborn |
| Deployment | Docker, Render.com |
| Tests | pytest |
