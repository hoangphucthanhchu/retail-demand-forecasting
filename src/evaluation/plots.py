"""
Training / evaluation figures for display (e.g. Jupyter via ``display_training_report_plots``).

Do not force the Agg backend at import time so notebooks can use an interactive
inline backend and ``plt.show()`` / IPython ``display(fig)`` work correctly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)

# Matches ``src.evaluation.baselines.evaluate_baselines`` column naming.
def _baseline_specs(target_col: str) -> List[Tuple[str, str]]:
    return [
        (f"{target_col}_lag_1", "Baseline — naive lag-1"),
        (f"{target_col}_lag_7", "Baseline — seasonal lag-7"),
        (f"{target_col}_rolling_mean_7", "Baseline — rolling mean 7d"),
    ]


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
    include_baselines: bool = True,
) -> Optional[Figure]:
    if date_col not in test_df.columns or target_col not in test_df.columns:
        logger.warning("Missing %s or %s; skipping actual vs predicted plot", date_col, target_col)
        return None
    X = test_df[feature_cols].fillna(0)
    frame = test_df[[date_col, target_col]].copy()
    baseline_cols: List[str] = []
    if include_baselines:
        for col, _ in _baseline_specs(target_col):
            if col in test_df.columns:
                frame[col] = test_df[col].fillna(0)
                baseline_cols.append(col)
    for name in model_trainer.models.keys():
        frame[f"pred_{name}"] = model_trainer.predict(name, X)
    pred_cols = [c for c in frame.columns if c.startswith("pred_")]
    if not pred_cols and not baseline_cols:
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
    baseline_colors = ["#7f7f7f", "#bcbd22", "#17becf"]
    spec_by_col = dict(_baseline_specs(target_col))
    for i, col in enumerate(baseline_cols):
        ax.plot(
            daily[date_col],
            daily[col],
            label=spec_by_col.get(col, col),
            color=baseline_colors[i % len(baseline_colors)],
            linestyle="--",
            linewidth=1.5,
            alpha=0.95,
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
    title = "Test set — daily total actual vs predicted"
    if include_baselines and baseline_cols:
        title += " (with baselines)"
    ax.set_title(title)
    ax.set_xlabel(date_col)
    ax.set_ylabel("Sum over rows")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def figure_sample_series_forecast(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_trainer: "ModelTrainer",
    *,
    model_name: str,
    date_col: str = "date",
    target_col: str = "demand",
    id_col: str = "id",
    n_series: int = 3,
) -> Optional[Figure]:
    """Actual vs predicted for a few series (rows), one chosen ML model."""
    need = {date_col, target_col, id_col}
    if not need.issubset(test_df.columns):
        logger.warning("Missing columns for sample-series plot; need %s", need)
        return None
    if model_name not in model_trainer.models:
        logger.warning("Model %s not found; skipping sample-series plot", model_name)
        return None
    if n_series <= 0:
        return None
    ids = test_df[id_col].unique()[:n_series]
    if len(ids) == 0:
        return None
    n = len(ids)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])
    X_all = test_df[feature_cols].fillna(0)
    y_hat = model_trainer.predict(model_name, X_all)
    work = test_df[[id_col, date_col, target_col]].copy()
    work["_pred"] = y_hat
    for ax, sid in zip(axes, ids):
        sub = work[work[id_col] == sid].sort_values(date_col)
        ax.plot(sub[date_col], sub[target_col], color="black", linewidth=1.8, label="Actual")
        ax.plot(sub[date_col], sub["_pred"], color="tab:blue", linewidth=1.4, alpha=0.9, label=f"Predicted ({model_name})")
        ax.set_ylabel("Demand")
        ax.set_title(f"{id_col} = {sid}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(date_col)
    fig.suptitle(f"Test set — sample series (model: {model_name})", y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def figure_residual_scatter(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_trainer: "ModelTrainer",
    *,
    model_name: str,
    target_col: str = "demand",
    max_points: int = 50_000,
    random_seed: int = 42,
) -> Optional[Figure]:
    """Actual vs predicted scatter on the test set (subsampled if large)."""
    if target_col not in test_df.columns:
        return None
    if model_name not in model_trainer.models:
        logger.warning("Model %s not found; skipping residual scatter", model_name)
        return None
    y_true = test_df[target_col].astype(float).values
    X = test_df[feature_cols].fillna(0)
    y_pred = model_trainer.predict(model_name, X).astype(float)
    n = len(y_true)
    if n > max_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n, size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    if hi <= lo:
        hi = lo + 1.0
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_pred, y_true, s=8, alpha=0.25, c="steelblue", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect fit")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Predicted ({model_name})")
    ax.set_ylabel(f"Actual ({target_col})")
    ax.set_title("Test set — actual vs predicted (per row)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


ForecastModelsArg = Optional[Union[str, Sequence[str]]]


def _resolve_forecast_models(
    model_trainer: "ModelTrainer",
    forecast_models: ForecastModelsArg,
) -> List[str]:
    """Sample-series and residual figures: ``None`` = all trained models; str = one; sequence = subset."""
    if forecast_models is None:
        return list(model_trainer.models.keys())
    names = [forecast_models] if isinstance(forecast_models, str) else list(forecast_models)
    if not names:
        return []
    out = [n for n in names if n in model_trainer.models]
    unknown = set(names) - set(out)
    if unknown:
        logger.warning("Unknown forecast_models (skipped): %s", unknown)
    return out


def iter_training_report_figures(
    *,
    test_results: Dict[str, Dict[str, float]],
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_trainer: "ModelTrainer",
    date_col: str = "date",
    target_col: str = "demand",
    top_n_features: int = 20,
    include_baselines_daily: bool = True,
    sample_series_n: int = 3,
    sample_id_col: str = "id",
    residual_scatter: bool = True,
    residual_max_points: int = 50_000,
    forecast_models: ForecastModelsArg = None,
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
        include_baselines=include_baselines_daily,
    )
    if fig is not None:
        yield fig

    for m in _resolve_forecast_models(model_trainer, forecast_models):
        if sample_series_n > 0:
            fig = figure_sample_series_forecast(
                test_df,
                feature_cols,
                model_trainer,
                model_name=m,
                date_col=date_col,
                target_col=target_col,
                id_col=sample_id_col,
                n_series=sample_series_n,
            )
            if fig is not None:
                yield fig

        if residual_scatter:
            fig = figure_residual_scatter(
                test_df,
                feature_cols,
                model_trainer,
                model_name=m,
                target_col=target_col,
                max_points=residual_max_points,
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
    include_baselines_daily: bool = True,
    sample_series_n: int = 3,
    sample_id_col: str = "id",
    residual_scatter: bool = True,
    residual_max_points: int = 50_000,
    forecast_models: ForecastModelsArg = None,
) -> None:
    """
    Show the standard post-training figures in a Jupyter notebook (or ``plt.show`` fallback).

    Daily chart can overlay the same lag/rolling baselines as ``evaluate_baselines``.
    ``forecast_models``: ``None`` = every trained model; ``"lightgbm"`` = one model;
    or a sequence such as ``("xgboost", "lightgbm")`` / ``("lightgbm",)``.
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
        include_baselines_daily=include_baselines_daily,
        sample_series_n=sample_series_n,
        sample_id_col=sample_id_col,
        residual_scatter=residual_scatter,
        residual_max_points=residual_max_points,
        forecast_models=forecast_models,
    ):
        if ipython_display is not None:
            ipython_display(fig)
        else:
            plt.show()
        plt.close(fig)
