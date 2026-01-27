"""
Evaluation metrics for demand forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error"""
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return (numerator / denominator) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     metrics: Optional[list] = None) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metric names to calculate
        
    Returns:
        Dictionary with metric names and values
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'mape', 'wmape']
    
    results = {}
    
    for metric in metrics:
        if metric == 'rmse':
            results['rmse'] = rmse(y_true, y_pred)
        elif metric == 'mae':
            results['mae'] = mae(y_true, y_pred)
        elif metric == 'mape':
            results['mape'] = mape(y_true, y_pred)
        elif metric == 'wmape':
            results['wmape'] = wmape(y_true, y_pred)
        else:
            logger.warning(f"Unknown metric: {metric}")
    
    return results


class Evaluator:
    """Evaluator for demand forecasting models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.metrics_list = self.config.get('evaluation', {}).get('metrics', ['rmse', 'mae', 'mape', 'wmape'])
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        return calculate_metrics(y_true, y_pred, self.metrics_list)
    
    def evaluate_by_group(self, df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                          group_col: str = 'id') -> pd.DataFrame:
        """
        Evaluate predictions grouped by a column (e.g., by product ID)
        
        Args:
            df: DataFrame with true and predicted values
            y_true_col: Column name for true values
            y_pred_col: Column name for predicted values
            group_col: Column to group by
            
        Returns:
            DataFrame with metrics per group
        """
        results = []
        
        for group, group_df in df.groupby(group_col):
            y_true = group_df[y_true_col].values
            y_pred = group_df[y_pred_col].values
            
            metrics = calculate_metrics(y_true, y_pred, self.metrics_list)
            metrics[group_col] = group
            results.append(metrics)
        
        return pd.DataFrame(results)
