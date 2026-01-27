"""
Tests for evaluation metrics
"""
import pytest
import numpy as np

from src.evaluation.metrics import rmse, mae, mape, wmape, calculate_metrics


def test_rmse():
    """Test RMSE calculation"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
    
    result = rmse(y_true, y_pred)
    assert result > 0
    assert isinstance(result, float)


def test_mae():
    """Test MAE calculation"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
    
    result = mae(y_true, y_pred)
    assert result > 0
    assert isinstance(result, float)


def test_calculate_metrics():
    """Test calculate_metrics function"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
    
    metrics = calculate_metrics(y_true, y_pred, metrics=['rmse', 'mae'])
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert isinstance(metrics['rmse'], float)
    assert isinstance(metrics['mae'], float)
