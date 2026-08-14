import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, split_data
from src.feature_engineering import FeatureEngineer
from src.models.baseline import BaselineModel
from src.models.xgboost_model import XGBoostModel
from src import evaluator
from src.explainer import shap_explainer
from report.generate_pdf import generate_report

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'insurance_claims.csv')

st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")


@st.cache_resource(show_spinner="Training models — this takes ~60s on first run...")
def load_and_train():
    df = load_data(DATA_PATH)
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

    return df, fe_baseline, fe_xgb, X_train_enc, X_val_sc, X_val_enc, y_val, baseline, xgb


def risk_label(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    return "HIGH"


def risk_color(label: str) -> str:
    return {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[label]


if not os.path.exists(DATA_PATH):
    st.error(
        "Dataset not found. Download `insurance_claims.csv` from "
        "https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data "
        "and place it at `data/insurance_claims.csv`."
    )
    st.stop()

df, fe_baseline, fe_xgb, X_train_enc, X_val_sc, X_val_enc, y_val, baseline, xgb = load_and_train()

page = st.sidebar.selectbox(
    "Navigation",
    ["1. Overview", "2. Model Evaluation", "3. Explainability", "4. Claim Risk Scorer"],
)

# ─── Page 1: Overview ────────────────────────────────────────────────────────
if page == "1. Overview":
    st.title("Insurance Fraud Claims Detection Engine")
    st.markdown("### Dataset Overview")

    fraud_count = int(df['fraud_reported'].sum())
    legit_count = len(df) - fraud_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims", f"{len(df):,}")
    col2.metric("Fraudulent", f"{fraud_count:,} ({fraud_count / len(df) * 100:.1f}%)")
    col3.metric("Legitimate", f"{legit_count:,} ({legit_count / len(df) * 100:.1f}%)")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Fraud vs Legitimate Distribution**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [legit_count, fraud_count],
            labels=['Legitimate', 'Fraud'],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            startangle=90,
        )
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("**Total Claim Amount by Outcome**")
        fig, ax = plt.subplots(figsize=(5, 4))
        df[df['fraud_reported'] == 0]['total_claim_amount'].plot(
            kind='hist', bins=30, alpha=0.6, label='Legitimate', ax=ax, color='#4CAF50'
        )
        df[df['fraud_reported'] == 1]['total_claim_amount'].plot(
            kind='hist', bins=30, alpha=0.6, label='Fraud', ax=ax, color='#F44336'
        )
        ax.set_xlabel('Total Claim Amount ($)')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("**Sample Records**")
    st.dataframe(df.head(10), use_container_width=True)


# ─── Page 2: Model Evaluation ────────────────────────────────────────────────
elif page == "2. Model Evaluation":
    st.title("Model Evaluation")

    class _ProxyBaseline:
        def predict_proba(self, _):
            return baseline.predict_proba(X_val_sc)

    class _ProxyXGB:
        def predict_proba(self, _):
            return xgb.predict_proba(X_val_enc)

    models_proxy = {
        'Logistic Regression': _ProxyBaseline(),
        'XGBoost': _ProxyXGB(),
    }

    metrics_df = evaluator.compare_models(models_proxy, None, y_val)

    st.markdown("### Metrics Comparison")
    st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ROC Curves**")
        fig_roc = evaluator.plot_roc_curves(models_proxy, None, y_val)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    with col2:
        st.markdown("**Precision-Recall Curves**")
        fig_pr = evaluator.plot_pr_curves(models_proxy, None, y_val)
        st.pyplot(fig_pr)
        plt.close(fig_pr)

    st.markdown("---")
    st.markdown("**Confusion Matrix (XGBoost, threshold = 0.5)**")
    xgb_proba_val = xgb.predict_proba(X_val_enc)
    fig_cm = evaluator.plot_confusion_matrix(y_val, xgb_proba_val)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.markdown("---")
    st.markdown("### Download PDF Report")
    if st.button("Generate PDF Report"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name

        dataset_stats = {
            'total_rows': len(df),
            'fraud_count': int(df['fraud_reported'].sum()),
            'fraud_pct': df['fraud_reported'].mean() * 100,
            'legit_count': int((df['fraud_reported'] == 0).sum()),
        }

        fig_roc2 = evaluator.plot_roc_curves(models_proxy, None, y_val)
        fig_cm2 = evaluator.plot_confusion_matrix(y_val, xgb_proba_val)
        fig_shap2 = shap_explainer.plot_beeswarm(xgb, X_train_enc, max_display=15)

        generate_report(tmp_path, dataset_stats, metrics_df, fig_roc2, fig_cm2, fig_shap2)
        plt.close('all')

        with open(tmp_path, 'rb') as f:
            st.download_button(
                label="Download Report PDF",
                data=f.read(),
                file_name="insurance_fraud_report.pdf",
                mime="application/pdf",
            )


# ─── Page 3: Explainability ──────────────────────────────────────────────────
elif page == "3. Explainability":
    st.title("Model Explainability (SHAP)")

    st.markdown("### Global Feature Importance — Beeswarm Plot")
    st.markdown("Shows which features push predictions toward fraud (red) or legitimate (blue).")

    with st.spinner("Computing SHAP values..."):
        sample_size = min(500, len(X_train_enc))
        X_sample = X_train_enc.sample(sample_size, random_state=42)
        fig_bee = shap_explainer.plot_beeswarm(xgb, X_sample, max_display=15)
    st.pyplot(fig_bee)
    plt.close(fig_bee)

    st.markdown("---")
    st.markdown("### Local Explanation — Single Claim Waterfall")
    claim_idx = st.slider("Select validation claim index", 0, len(X_val_enc) - 1, 0)
    X_single = X_val_enc.iloc[[claim_idx]]
    actual_label = "Fraud" if int(y_val.iloc[claim_idx]) == 1 else "Legitimate"
    pred_prob = xgb.predict_proba(X_single)[0]
    label = risk_label(pred_prob)

    st.markdown(
        f"**Actual:** {actual_label} &nbsp;|&nbsp; "
        f"**Predicted Fraud Probability:** {pred_prob:.3f} &nbsp;|&nbsp; "
        f"**Risk:** :{risk_color(label)}[{label}]"
    )

    with st.spinner("Computing waterfall..."):
        fig_wf = shap_explainer.plot_waterfall(xgb, X_single)
    st.pyplot(fig_wf)
    plt.close(fig_wf)


# ─── Page 4: Claim Risk Scorer ───────────────────────────────────────────────
elif page == "4. Claim Risk Scorer":
    st.title("Claim Risk Scorer")
    st.markdown("Enter claim details to get a fraud probability and explanation.")

    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            months_as_customer = st.number_input("Months as Customer", 0, 600, 120)
            age = st.number_input("Age", 18, 90, 35)
            policy_deductable = st.selectbox("Policy Deductable ($)", [500, 1000, 2000])
            policy_annual_premium = st.number_input("Annual Premium ($)", 500.0, 5000.0, 1200.0)
            umbrella_limit = st.number_input("Umbrella Limit ($)", 0, 10000000, 0, step=1000000)
            insured_sex = st.selectbox("Insured Sex", ['MALE', 'FEMALE'])
            insured_education_level = st.selectbox(
                "Education Level",
                ['MD', 'PhD', 'Associate', 'JD', 'College', 'Masters', 'High School'],
            )
            insured_occupation = st.selectbox(
                "Occupation",
                ['craft-repair', 'exec-managerial', 'farming-fishing', 'handlers-cleaners',
                 'machine-op-inspct', 'other-service', 'priv-house-serv', 'prof-specialty',
                 'protective-serv', 'sales', 'tech-support', 'transport-moving', 'armed-forces'],
            )

        with col2:
            insured_relationship = st.selectbox(
                "Relationship",
                ['husband', 'wife', 'own-child', 'not-in-family', 'other-relative', 'unmarried'],
            )
            incident_type = st.selectbox(
                "Incident Type",
                ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Parked Car', 'Vehicle Theft'],
            )
            collision_type = st.selectbox(
                "Collision Type",
                ['Front Collision', 'Rear Collision', 'Side Collision', 'NA'],
            )
            incident_severity = st.selectbox(
                "Incident Severity",
                ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'],
            )
            incident_hour = st.slider("Incident Hour (0-23)", 0, 23, 14)
            num_vehicles = st.number_input("Vehicles Involved", 1, 4, 1)
            bodily_injuries = st.number_input("Bodily Injuries", 0, 2, 0)

        with col3:
            witnesses = st.number_input("Witnesses", 0, 3, 1)
            police_report = st.selectbox("Police Report Available", ['YES', 'NO'])
            property_damage_val = st.selectbox("Property Damage", ['YES', 'NO'])
            total_claim = st.number_input("Total Claim Amount ($)", 0, 100000, 5000)
            injury_claim = st.number_input("Injury Claim ($)", 0, 50000, 1000)
            property_claim = st.number_input("Property Claim ($)", 0, 50000, 2000)
            vehicle_claim = st.number_input("Vehicle Claim ($)", 0, 50000, 2000)
            auto_make = st.selectbox(
                "Auto Make",
                ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Audi', 'Mercedes', 'Dodge',
                 'Volkswagen', 'Nissan', 'Subaru', 'Jeep', 'RAM', 'GMC', 'Accura', 'Saab'],
            )
            auto_year = st.number_input("Auto Year", 1995, 2015, 2010)

        submitted = st.form_submit_button("Score Claim")

    if submitted:
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
            'collision_type': collision_type if collision_type != 'NA' else None,
            'incident_severity': incident_severity,
            'incident_hour_of_the_day': incident_hour,
            'number_of_vehicles_involved': num_vehicles,
            'bodily_injuries': bodily_injuries,
            'witnesses': witnesses,
            'police_report_available': police_report,
            'property_damage': property_damage_val,
            'total_claim_amount': total_claim,
            'injury_claim': injury_claim,
            'property_claim': property_claim,
            'vehicle_claim': vehicle_claim,
            'auto_make': auto_make,
            'auto_year': auto_year,
        }])

        X_input = fe_xgb.transform(input_data)
        prob = xgb.predict_proba(X_input)[0]
        label = risk_label(prob)
        color = risk_color(label)

        st.markdown("---")
        st.markdown(f"### Fraud Probability: **{prob:.1%}**")
        st.markdown(f"### Risk Level: :{color}[**{label}**]")
        st.progress(float(prob))

        st.markdown("---")
        st.markdown("**Why this prediction? (SHAP Waterfall)**")
        with st.spinner("Computing explanation..."):
            fig_wf = shap_explainer.plot_waterfall(xgb, X_input)
        st.pyplot(fig_wf)
        plt.close(fig_wf)
