import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def raw_X():
    return pd.DataFrame({
        'months_as_customer': [12, 24, 36],
        'age': [25, 35, 45],
        'policy_deductable': [500, 1000, 500],
        'policy_annual_premium': [1200.0, 1500.0, 1100.0],
        'umbrella_limit': [0, 1000000, 0],
        'insured_sex': ['MALE', 'FEMALE', 'MALE'],
        'insured_education_level': ['MD', 'PhD', 'Associate'],
        'insured_occupation': ['craft-repair', 'exec-managerial', 'sales'],
        'insured_relationship': ['husband', 'own-child', 'wife'],
        'incident_type': ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Parked Car'],
        'collision_type': ['Front Collision', None, None],
        'incident_severity': ['Major Damage', 'Minor Damage', 'Trivial Damage'],
        'incident_hour_of_the_day': [2, 14, 23],
        'number_of_vehicles_involved': [1, 2, 1],
        'bodily_injuries': [0, 1, 0],
        'witnesses': [0, 1, 2],
        'police_report_available': ['NO', 'YES', 'NO'],
        'property_damage': ['YES', None, 'NO'],
        'total_claim_amount': [5000, 10000, 2000],
        'injury_claim': [1000, 3000, 500],
        'property_claim': [2000, 4000, 1000],
        'vehicle_claim': [2000, 3000, 500],
        'auto_make': ['Toyota', 'Honda', 'Ford'],
        'auto_year': [2010, 2015, 2012],
    })


def test_fit_transform_creates_derived_features(raw_X):
    fe = FeatureEngineer()
    out = fe.fit_transform(raw_X)
    assert 'claim_to_premium_ratio' in out.columns
    assert 'is_night_incident' in out.columns
    assert 'no_police_no_witness' in out.columns


def test_is_night_incident_values(raw_X):
    fe = FeatureEngineer()
    out = fe.fit_transform(raw_X)
    # hour=2 → night, hour=14 → day, hour=23 → night
    assert list(out['is_night_incident']) == [1, 0, 1]


def test_no_police_no_witness(raw_X):
    fe = FeatureEngineer()
    out = fe.fit_transform(raw_X)
    # row 0: police=NO, witnesses=0 → 1; row 1: police=YES → 0; row 2: police=NO, witnesses=2 → 0
    assert out['no_police_no_witness'].iloc[0] == 1
    assert out['no_police_no_witness'].iloc[1] == 0
    assert out['no_police_no_witness'].iloc[2] == 0


def test_fit_transform_no_nulls(raw_X):
    fe = FeatureEngineer()
    out = fe.fit_transform(raw_X)
    assert out.isnull().sum().sum() == 0


def test_transform_aligns_columns(raw_X):
    fe = FeatureEngineer()
    fe.fit_transform(raw_X)
    out = fe.transform(raw_X)
    assert list(out.columns) == fe.encoded_columns


def test_fit_transform_scaled_no_nulls(raw_X):
    fe = FeatureEngineer()
    out = fe.fit_transform_scaled(raw_X)
    assert out.isnull().sum().sum() == 0
