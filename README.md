# Insurance Fraud Claims Detection Engine

A machine learning prototype that detects fraudulent auto insurance claims using Logistic Regression and XGBoost, with SHAP explainability and a PDF case study report.

## Live Demo
Deployed on Streamlit Community Cloud.

## Features
- **Dataset**: 1,000 auto insurance claims with fraud labels
- **Models**: Logistic Regression (baseline) vs XGBoost (tuned with GridSearch)
- **Explainability**: SHAP beeswarm (global) + waterfall (per-claim)
- **4-Page UI**: Overview · Model Evaluation · Explainability · Claim Risk Scorer
- **PDF Report**: Downloadable case study with charts

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
