"""
Training / evaluation figures for display (e.g. Jupyter via ``display_training_report_plots``).

Do not force the Agg backend at import time so notebooks can use an interactive
inline backend and ``plt.show()`` / IPython ``display(fig)`` work correctly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def _eval_curve_from_booster_model(model: Any) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Read validation metric trajectory from a fitted XGBoost / LightGBM sklearn estimator.
    """
    er = getattr(model, "evals_result_", None)
    if not er or not isinstance(er, dict):
        return None
    fold_key = next(iter(er), None)
    if fold_key is None:
        return None
    fold = er[fold_key]
    if not isinstance(fold, dict):
        return None
    metric_key = next(iter(fold), None)
    if metric_key is None:
        return None
    values = fold[metric_key]
    if values is None:
        return None
    ys = np.asarray(values, dtype=float)
    xs = np.arange(1, len(ys) + 1)
    return xs, ys, f"{metric_key} ({fold_key})"


def figure_metrics_comparison(test_results: Dict[str, Dict[str, float]]) -> Optional[Figure]:
    if not test_results:
        return None
    df = pd.DataFrame(test_results).T
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind="bar", ax=ax, rot=45)
    ax.set_title("Test set — metric comparison")
    ax.set_ylabel("Value")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def figure_learning_curve(model_name: str, model: Any) -> Optional[Figure]:
    series = _eval_curve_from_booster_model(model)
    if series is None:
        logger.warning("No eval history for model %s; skipping learning curve", model_name)
        return None
    xs, ys, ylabel = series
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, color="steelblue", linewidth=1.5)
    ax.set_title(f"Validation metric during training — {model_name}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def figure_feature_importance(
    model_trainer: "ModelTrainer",
    model_name: str,
    top_n: int = 20,
) -> Optional[Figure]:
    try:
        imp = model_trainer.get_feature_importance(model_name, top_n=top_n)
    except ValueError:
        return None
    if imp.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)))
    ax.barh(imp["feature"], imp["importance"], color="teal", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features — {model_name}")
    fig.tight_layout()
    return fig


def figure_actual_vs_predicted_daily(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_trainer: "ModelTrainer",
    date_col: str = "date",
    target_col: str = "demand",
) -> Optional[Figure]:
    if date_col not in test_df.columns or target_col not in test_df.columns:
        logger.warning("Missing %s or %s; skipping actual vs predicted plot", date_col, target_col)
        return None
    X = test_df[feature_cols].fillna(0)
    frame = test_df[[date_col, target_col]].copy()
    for name in model_trainer.models.keys():
        frame[f"pred_{name}"] = model_trainer.predict(name, X)
    pred_cols = [c for c in frame.columns if c.startswith("pred_")]
    if not pred_cols:
        return None
    daily = frame.groupby(date_col, as_index=False).sum(numeric_only=True)
    daily = daily.sort_values(date_col)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        daily[date_col],
        daily[target_col],
        label="Actual (daily sum)",
        color="black",
        linewidth=2,
    )
    colors = plt.cm.tab10.colors
    for i, col in enumerate(pred_cols):
        label = col.replace("pred_", "Predicted — ")
        ax.plot(
            daily[date_col],
            daily[col],
            label=label,
            color=colors[i % len(colors)],
            alpha=0.9,
        )
    ax.set_title("Test set — daily total actual vs predicted")
    ax.set_xlabel(date_col)
    ax.set_ylabel("Sum over rows")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def iter_training_report_figures(
    *,
    test_results: Dict[str, Dict[str, float]],
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_trainer: "ModelTrainer",
    date_col: str = "date",
    target_col: str = "demand",
    top_n_features: int = 20,
) -> Iterator[Figure]:
    """Yield each figure in the standard post-training report (caller should ``close`` after use)."""
    fig = figure_metrics_comparison(test_results)
    if fig is not None:
        yield fig

    for name, model in model_trainer.models.items():
        fig = figure_learning_curve(name, model)
        if fig is not None:
            yield fig

    for name in model_trainer.models.keys():
        fig = figure_feature_importance(model_trainer, name, top_n=top_n_features)
        if fig is not None:
            yield fig

    fig = figure_actual_vs_predicted_daily(
        test_df,
        feature_cols,
        model_trainer,
        date_col=date_col,
        target_col=target_col,
    )
    if fig is not None:
        yield fig


def display_training_report_plots(
    *,
    test_results: Dict[str, Dict[str, float]],
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_trainer: "ModelTrainer",
    date_col: str = "date",
    target_col: str = "demand",
    top_n_features: int = 20,
) -> None:
    """
    Show the standard post-training figures in a Jupyter notebook (or ``plt.show`` fallback).
    """
    try:
        from IPython.display import display as ipython_display
    except ImportError:
        ipython_display = None

    for fig in iter_training_report_figures(
        test_results=test_results,
        test_df=test_df,
        feature_cols=feature_cols,
        model_trainer=model_trainer,
        date_col=date_col,
        target_col=target_col,
        top_n_features=top_n_features,
    ):
        if ipython_display is not None:
            ipython_display(fig)
        else:
            plt.show()
        plt.close(fig)
