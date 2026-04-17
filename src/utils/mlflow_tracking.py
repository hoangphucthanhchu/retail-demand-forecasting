"""
MLflow experiment tracking helpers.

Tracking is gated by ``config["mlflow"]["enabled"]`` so existing configs without
an ``mlflow`` block leave behavior unchanged.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional

from src.utils.config import infer_project_root

logger = logging.getLogger(__name__)


def is_mlflow_enabled(config: Mapping[str, Any]) -> bool:
    ml = config.get("mlflow")
    if not isinstance(ml, dict):
        return False
    return bool(ml.get("enabled", False))


def resolved_local_tracking_uri(
    config: Mapping[str, Any],
    config_path: Optional[str],
) -> str:
    """
    Fallback when ``mlflow.tracking_uri`` is unset: ``<parent(models_dir)>/mlruns``,
    else ``<project_root>/mlruns`` from ``config_path``, else ``<cwd>/mlruns``.
    """
    paths = config.get("paths") or {}
    md = paths.get("models_dir")
    if md:
        root = Path(md).resolve().parent
        return (root / "mlruns").resolve().as_uri()

    if config_path:
        root = infer_project_root(config_path)
        return (root / "mlruns").resolve().as_uri()

    return Path("mlruns").resolve().as_uri()


def _safe_metric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        import math

        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _mlflow_active() -> bool:
    try:
        import mlflow
    except ImportError:
        return False
    return mlflow.active_run() is not None


def log_nested_metrics(prefix: str, results: Mapping[str, Mapping[str, Any]]) -> None:
    """Log ``results[model_or_baseline][metric_name]`` as ``{prefix}_{name}_{metric}``."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed; skipping metric logging")
        return
    if not _mlflow_active():
        return

    for name, metrics in results.items():
        safe = str(name).replace(" ", "_").replace("/", "_")
        for metric_name, raw in metrics.items():
            v = _safe_metric_value(raw)
            if v is None:
                continue
            mlflow.log_metric(f"{prefix}_{safe}_{metric_name}", v)


def log_models_from_trainer(model_trainer: Any, models_dir: str) -> None:
    """Register each trained estimator with MLflow using native flavors when possible."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed; skipping model logging")
        return

    if not _mlflow_active():
        return

    models_path = Path(models_dir)

    for name, model in model_trainer.models.items():
        artifact_subpath = f"models/{name}"
        pkl = models_path / f"{name}_model.pkl"
        try:
            if name == "xgboost":
                import mlflow.xgboost

                mlflow.xgboost.log_model(model, artifact_subpath)
            elif name == "lightgbm":
                import mlflow.lightgbm

                mlflow.lightgbm.log_model(model, artifact_subpath)
            else:
                import mlflow.sklearn

                mlflow.sklearn.log_model(model, artifact_subpath)
            logger.info(
                "MLflow: logged model flavor under run artifacts → %s (see UI: Experiments → run → Artifacts)",
                artifact_subpath,
            )
        except Exception as e:
            logger.warning("MLflow log_model failed for %s (%s); logging joblib artifact", name, e)
            if pkl.is_file():
                mlflow.log_artifact(str(pkl), artifact_path="joblib_checkpoints")
        if pkl.is_file():
            try:
                mlflow.log_artifact(str(pkl), artifact_path="pickles")
            except Exception as e:
                logger.warning("MLflow: could not attach pickle for %s: %s", name, e)


def log_config_snapshot(config: MutableMapping[str, Any]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    if not _mlflow_active():
        return
    # Resolved paths etc.; keep reproducibility of the run as seen by the pipeline.
    try:
        mlflow.log_dict(dict(config), "config_resolved.json")
    except Exception as e:
        logger.warning("Could not log config snapshot to MLflow: %s", e)


def log_training_tags(
    config: Mapping[str, Any],
    *,
    config_path: Optional[str],
    n_features: int,
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    try:
        import mlflow
    except ImportError:
        return
    if not _mlflow_active():
        return

    mlflow.set_tag("task", "demand_forecast")
    if config_path:
        mlflow.set_tag("config_path", config_path)
        name = Path(config_path).name
        mlflow.set_tag("config_name", name)
        mlflow.log_param("config_name", name)
    mlflow.set_tag("n_features", str(n_features))
    mlflow.set_tag("n_train_rows", str(n_train))
    mlflow.set_tag("n_val_rows", str(n_val))
    mlflow.set_tag("n_test_rows", str(n_test))
    subsample = (config.get("data") or {}).get("subsample") or {}
    if isinstance(subsample, dict) and subsample.get("enabled") is not None:
        mlflow.set_tag("data_subsample_enabled", str(subsample.get("enabled")))


def _flatten_params(d: Mapping[str, Any], prefix: str = "") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, Mapping):
            out.update(_flatten_params(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        elif v is None:
            out[key] = "null"
        elif isinstance(v, bool):
            out[key] = str(v).lower()
        elif isinstance(v, (int, float, str)):
            out[key] = str(v)
        else:
            out[key] = str(v)
    return out


def log_searchable_params(config: Mapping[str, Any]) -> None:
    """Log a flattened subset of hyperparameters (MLflow param count limits)."""
    try:
        import mlflow
    except ImportError:
        return
    if not _mlflow_active():
        return

    model_cfg = config.get("model") or {}
    train_cfg = config.get("training") or {}
    flat = _flatten_params({"model": model_cfg, "training": train_cfg})
    # MLflow allows at most 100 params per batch in some versions; chunk safely.
    items = list(flat.items())[:100]
    batch: Dict[str, str] = {}
    for i, (k, val) in enumerate(items):
        batch[k] = val[:250]  # param value length cap
        if len(batch) >= 50 or i == len(items) - 1:
            mlflow.log_params(batch)
            batch = {}


@contextmanager
def active_run(
    config: Mapping[str, Any],
    *,
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Iterator[None]:
    """
    Context manager: sets tracking URI and experiment, then ``mlflow.start_run``.

    When ``mlflow.tracking_uri`` is empty, uses :func:`resolved_local_tracking_uri`.
    Prefer ``tracking_uri: "file:./mlruns"`` in YAML; ``load_config`` resolves it to
    an absolute ``file://`` URI under the project root.

    Yields nothing when MLflow is disabled or not installed.
    """
    if not is_mlflow_enabled(config):
        yield
        return

    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed; continuing without experiment tracking")
        yield
        return

    ml = config.get("mlflow") or {}
    uri = (ml.get("tracking_uri") or "").strip()
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        local_uri = resolved_local_tracking_uri(config, config_path)
        mlflow.set_tracking_uri(local_uri)
        logger.info("MLflow: using unified local store %s", local_uri)

    experiment_name = ml.get("experiment_name") or "retail-demand-forecasting"
    experiment = mlflow.set_experiment(experiment_name)

    run_kw: Dict[str, Any] = {"experiment_id": experiment.experiment_id}
    if run_name:
        run_kw["run_name"] = run_name

    with mlflow.start_run(**run_kw):
        yield
