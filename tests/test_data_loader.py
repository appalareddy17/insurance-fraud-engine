import pandas as pd
import pytest
from src.data_loader import load_data, split_data


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'months_as_customer': [12, 24, 36, 48, 60, 72, 84],
        'age': [25, 35, 45, 30, 55, 28, 40],
        'policy_deductable': [500, 1000, 500, 2000, 1000, 500, 2000],
        'policy_annual_premium': [1200.0, 1500.0, 1100.0, 1800.0, 1300.0, 1000.0, 1600.0],
        'umbrella_limit': [0, 1000000, 0, 2000000, 0, 0, 1000000],
        'insured_sex': ['MALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE'],
        'insured_education_level': ['MD', 'PhD', 'Associate', 'MD', 'JD', 'College', 'PhD'],
        'insured_occupation': ['craft-repair', 'exec-managerial', 'sales', 'tech-support', 'craft-repair', 'sales', 'exec-managerial'],
        'insured_relationship': ['husband', 'own-child', 'wife', 'unmarried', 'husband', 'own-child', 'wife'],
        'incident_type': ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Parked Car', 'Vehicle Theft', 'Single Vehicle Collision', 'Multi-vehicle Collision', 'Parked Car'],
        'collision_type': ['Front Collision', 'Rear Collision', None, None, 'Side Collision', 'Rear Collision', None],
        'incident_severity': ['Major Damage', 'Minor Damage', 'Trivial Damage', 'Total Loss', 'Major Damage', 'Minor Damage', 'Trivial Damage'],
        'incident_hour_of_the_day': [14, 3, 11, 22, 8, 1, 16],
        'number_of_vehicles_involved': [1, 2, 1, 1, 1, 3, 1],
        'bodily_injuries': [0, 1, 0, 0, 2, 1, 0],
        'witnesses': [2, 0, 1, 0, 3, 0, 2],
        'police_report_available': ['YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES'],
        'property_damage': ['YES', 'NO', None, 'YES', 'NO', None, 'YES'],
        'total_claim_amount': [5000, 10000, 2000, 15000, 8000, 12000, 3000],
        'injury_claim': [1000, 3000, 500, 5000, 2000, 4000, 800],
        'property_claim': [2000, 4000, 1000, 6000, 3000, 5000, 1200],
        'vehicle_claim': [2000, 3000, 500, 4000, 3000, 3000, 1000],
        'auto_make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Toyota', 'Honda', 'Ford'],
        'auto_year': [2010, 2015, 2012, 2018, 2008, 2014, 2011],
        'fraud_reported': ['Y', 'N', 'N', 'Y', 'N', 'Y', 'N'],
    })


def test_load_data_returns_correct_columns(tmp_path, sample_df):
    path = tmp_path / "test.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(str(path))
    assert 'fraud_reported' in df.columns
    assert set(df['fraud_reported'].unique()).issubset({0, 1})


def test_load_data_raises_on_missing_column(tmp_path, sample_df):
    bad_df = sample_df.drop(columns=['fraud_reported'])
    path = tmp_path / "bad.csv"
    bad_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_data(str(path))


def test_load_data_drops_id_columns(tmp_path, sample_df):
    sample_df['policy_number'] = range(7)
    sample_df['insured_zip'] = ['12345'] * 7
    path = tmp_path / "test.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(str(path))
    assert 'policy_number' not in df.columns
    assert 'insured_zip' not in df.columns


def test_split_data_proportions(sample_df):
    sample_df['fraud_reported'] = (sample_df['fraud_reported'] == 'Y').astype(int)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df)
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(sample_df)
    assert len(X_train) > len(X_val)


def test_split_data_no_target_in_X(sample_df):
    sample_df['fraud_reported'] = (sample_df['fraud_reported'] == 'Y').astype(int)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df)
    assert 'fraud_reported' not in X_train.columns
