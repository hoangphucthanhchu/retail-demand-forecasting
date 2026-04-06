"""
Simple forecast baselines for comparison (same metrics as ML models).
"""
from typing import Dict

import pandas as pd

from src.evaluation.metrics import Evaluator


def evaluate_baselines(
    df: pd.DataFrame,
    evaluator: Evaluator,
    target_col: str = "demand",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate baseline forecasts built from existing lag/rolling columns.

    Uses fillna(0) on predictions to match how tree models consume features.

    Returns:
        Maps baseline name -> metric dict (e.g. naive_lag1, seasonal_naive_lag7).
    """
    y_true = df[target_col].values
    results: Dict[str, Dict[str, float]] = {}

    specs = [
        ("baseline_naive_lag1", f"{target_col}_lag_1"),
        ("baseline_seasonal_naive_lag7", f"{target_col}_lag_7"),
        ("baseline_rolling_mean_7", f"{target_col}_rolling_mean_7"),
    ]

    for name, col in specs:
        if col not in df.columns:
            continue
        y_pred = df[col].fillna(0).values
        results[name] = evaluator.evaluate(y_true, y_pred)

    return results
