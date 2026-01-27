"""
Tests for feature engineering
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample time series data"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    ids = ['id_1', 'id_2']
    
    data = []
    for id_val in ids:
        for date in dates:
            data.append({
                'id': id_val,
                'date': date,
                'demand': np.random.randint(0, 100)
            })
    
    return pd.DataFrame(data)


def test_calendar_features(sample_data):
    """Test calendar feature creation"""
    fe = FeatureEngineer()
    df = fe.create_calendar_features(sample_data)
    
    assert 'day_of_week' in df.columns
    assert 'month' in df.columns
    assert 'is_weekend' in df.columns
    assert 'day_of_week_sin' in df.columns


def test_lag_features(sample_data):
    """Test lag feature creation"""
    fe = FeatureEngineer()
    df = fe.create_lag_features(sample_data, group_cols=['id'], target_col='demand')
    
    assert 'demand_lag_1' in df.columns
    assert 'demand_lag_7' in df.columns


def test_rolling_features(sample_data):
    """Test rolling feature creation"""
    fe = FeatureEngineer()
    df = fe.create_rolling_features(sample_data, group_cols=['id'], target_col='demand')
    
    assert 'demand_rolling_mean_7' in df.columns
    assert 'demand_rolling_std_7' in df.columns
