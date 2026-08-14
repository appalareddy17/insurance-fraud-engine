import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = [
    'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
    'bodily_injuries', 'witnesses', 'total_claim_amount',
    'injury_claim', 'property_claim', 'vehicle_claim', 'auto_year',
]

CATEGORICAL_COLS = [
    'insured_sex', 'insured_education_level', 'insured_occupation',
    'insured_relationship', 'incident_type', 'collision_type',
    'incident_severity', 'police_report_available', 'property_damage', 'auto_make',
]


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.medians: dict = {}
        self.modes: dict = {}
        self.encoded_columns: list = []

    def _add_derived(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['claim_to_premium_ratio'] = (
            X['total_claim_amount'] / X['policy_annual_premium'].replace(0, np.nan)
        ).fillna(0)
        X['is_night_incident'] = (
            (X['incident_hour_of_the_day'] < 6) | (X['incident_hour_of_the_day'] > 22)
        ).astype(int)
        police_no = X['police_report_available'].astype(str).str.upper().isin(['NO', 'N', 'NONE', 'NAN'])
        witness_zero = X['witnesses'] == 0
        X['no_police_no_witness'] = (police_no & witness_zero).astype(int)
        return X

    def _base_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in NUMERIC_COLS:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.medians.get(col, 0))
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(self.modes.get(col, 'UNKNOWN')).astype(str)
        X = self._add_derived(X)
        X = pd.get_dummies(X, columns=CATEGORICAL_COLS)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in NUMERIC_COLS:
            if col in X.columns:
                self.medians[col] = pd.to_numeric(X[col], errors='coerce').median()
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                mode_vals = X[col].dropna().mode()
                self.modes[col] = mode_vals[0] if len(mode_vals) > 0 else 'UNKNOWN'
        X_out = self._base_transform(X)
        self.encoded_columns = X_out.columns.tolist()
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = self._base_transform(X)
        for col in self.encoded_columns:
            if col not in X_out.columns:
                X_out[col] = 0
        return X_out[self.encoded_columns]

    def fit_transform_scaled(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = self.fit_transform(X)
        X_scaled = X_enc.copy()
        scale_cols = [c for c in NUMERIC_COLS + ['claim_to_premium_ratio'] if c in X_scaled.columns]
        X_scaled[scale_cols] = self.scaler.fit_transform(X_enc[scale_cols])
        return X_scaled

    def transform_scaled(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = self.transform(X)
        X_scaled = X_enc.copy()
        scale_cols = [c for c in NUMERIC_COLS + ['claim_to_premium_ratio'] if c in X_scaled.columns]
        X_scaled[scale_cols] = self.scaler.transform(X_enc[scale_cols])
        return X_scaled
