import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = [
    'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'insured_sex', 'insured_education_level', 'insured_occupation',
    'insured_relationship', 'incident_type', 'collision_type', 'incident_severity',
    'incident_hour_of_the_day', 'number_of_vehicles_involved', 'bodily_injuries',
    'witnesses', 'police_report_available', 'property_damage',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
    'auto_make', 'auto_year', 'fraud_reported',
]

DROP_COLUMNS = [
    'policy_number', 'incident_location', 'incident_date',
    'policy_bind_date', 'insured_zip',
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace('?', pd.NA)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=['fraud_reported'])
    df['fraud_reported'] = (df['fraud_reported'] == 'Y').astype(int)
    fraud_rate = df['fraud_reported'].mean()
    if fraud_rate < 0.09 or fraud_rate > 0.91:
        warnings.warn(f"Extreme class imbalance detected: fraud rate = {fraud_rate:.2%}")
    return df


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    X = df.drop(columns=['fraud_reported'])
    y = df['fraud_reported']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
    )
    val_ratio = val_size / (val_size + test_size)
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=random_state
        )
    except ValueError:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state
        )
    return X_train, X_val, X_test, y_train, y_val, y_test
