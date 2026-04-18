"""
MLflow experiment tracking helpers.

Tracking is gated by ``config["mlflow"]["enabled"]`` so existing configs without
an ``mlflow`` block leave behavior unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional

from src.utils.config import infer_project_root

logger = logging.getLogger(__name__)


def effective_feature_version(config: Mapping[str, Any]) -> str:
    """
    Human ``features.feature_version`` from config if set; otherwise a stable
    short fingerprint of the ``features`` block (excluding ``feature_version``).
    """
    feats = config.get("features") or {}
    if not isinstance(feats, dict):
        return "none"
    explicit = feats.get("feature_version")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    for_hash = {k: v for k, v in feats.items() if k != "feature_version"}
    canonical = json.dumps(for_hash, sort_keys=True, default=str)
    return "auto-" + hashlib.sha256(canonical.encode()).hexdigest()[:12]


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


def _model_uri_from_log_result(logged: Any, run_id: str, fallback_name: str) -> str:
    """Prefer URI returned by MLflow 3+ ``log_model`` (often ``models:/...``)."""
    if logged is None:
        return f"runs:/{run_id}/{fallback_name}"
    for attr in ("model_uri", "uri"):
        uri = getattr(logged, attr, None)
        if isinstance(uri, str) and uri.strip():
            return uri.strip()
    return f"runs:/{run_id}/{fallback_name}"


def _mlflow_log_model_name(model_key: str, *, suffix: str = "") -> str:
    """
    MLflow 3+ ``log_model(..., name=...)``: no ``/``, ``:``, ``.``, ``%``, quotes.
    """
    base = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{model_key}{suffix}".strip("_"))
    return (base or "model")[:200]


def _registry_safe_name(name: str) -> str:
    """Model Registry name: letters, digits, dash, underscore (no dots for parity with log_model rules)."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())
    return s[:200] if len(s) > 200 else s


def log_models_from_trainer(
    model_trainer: Any,
    models_dir: str,
    config: Optional[Mapping[str, Any]] = None,
    input_example: Optional[Any] = None,
) -> None:
    """
    Log each trained model as a run artifact; optionally register in the Model Registry.

    Pass ``input_example`` (e.g. a few training feature rows) so MLflow can infer a model
    signature and avoid registry URI mismatches on MLflow 3+.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed; skipping model logging")
        return

    if not _mlflow_active():
        return

    ml_cfg = (config or {}).get("mlflow") or {}
    do_register = bool(ml_cfg.get("register_models", False))
    prefix = (ml_cfg.get("registered_model_prefix") or "retail_demand").strip() or "retail_demand"

    models_path = Path(models_dir)
    run = mlflow.active_run()
    run_id = run.info.run_id if run else None

    try:
        store = mlflow.get_tracking_uri() or ""
        mlflow.set_tag("mlflow_tracking_uri", store[:500])
    except Exception:
        store = ""

    log_model_kw: Dict[str, Any] = {}
    if input_example is not None and getattr(input_example, "__len__", None):
        try:
            if len(input_example) > 0:
                log_model_kw["input_example"] = input_example
        except TypeError:
            log_model_kw["input_example"] = input_example

    for name, model in model_trainer.models.items():
        mlflow_name = _mlflow_log_model_name(str(name))
        register_path = mlflow_name
        model_uri_for_registry: Optional[str] = None
        pkl = models_path / f"{name}_model.pkl"
        flavor_ok = False
        try:
            if name == "xgboost":
                import mlflow.xgboost

                logged = mlflow.xgboost.log_model(model, name=mlflow_name, **log_model_kw)
                flavor_ok = True
                if run_id:
                    model_uri_for_registry = _model_uri_from_log_result(logged, run_id, mlflow_name)
            elif name == "lightgbm":
                import mlflow.lightgbm

                logged = mlflow.lightgbm.log_model(model, name=mlflow_name, **log_model_kw)
                flavor_ok = True
                if run_id:
                    model_uri_for_registry = _model_uri_from_log_result(logged, run_id, mlflow_name)
            else:
                import mlflow.sklearn

                logged = mlflow.sklearn.log_model(model, name=mlflow_name, **log_model_kw)
                flavor_ok = True
                if run_id:
                    model_uri_for_registry = _model_uri_from_log_result(logged, run_id, mlflow_name)
            logger.info(
                "MLflow: logged model under run Artifacts → %s (UI: Experiments → this run → Artifacts)",
                mlflow_name,
            )
        except Exception as e:
            logger.warning("MLflow native log_model failed for %s (%s)", name, e)
            alt = _mlflow_log_model_name(str(name), suffix="_sklearn")
            try:
                import mlflow.sklearn

                logged = mlflow.sklearn.log_model(model, name=alt, **log_model_kw)
                register_path = alt
                flavor_ok = True
                if run_id:
                    model_uri_for_registry = _model_uri_from_log_result(logged, run_id, alt)
                logger.info(
                    "MLflow: sklearn flavor logged for %s → %s (native flavor failed; model still loadable)",
                    name,
                    alt,
                )
            except Exception as e2:
                logger.warning("MLflow sklearn fallback log_model also failed for %s: %s", name, e2)
            if pkl.is_file():
                mlflow.log_artifact(str(pkl), artifact_path="joblib_checkpoints")
        if pkl.is_file():
            try:
                mlflow.log_artifact(str(pkl), artifact_path="pickles")
            except Exception as e:
                logger.warning("MLflow: could not attach pickle for %s: %s", name, e)

        if do_register and run_id:
            if not flavor_ok:
                logger.warning(
                    "MLflow: skip Model Registry for %s (no MLflow model flavor logged). "
                    "Install matching mlflow/xgboost/lightgbm or check errors above.",
                    name,
                )
            else:
                reg_name = _registry_safe_name(f"{prefix}_{name}")
                model_uri = model_uri_for_registry or f"runs:/{run_id}/{register_path}"
                try:
                    result = mlflow.register_model(model_uri=model_uri, name=reg_name)
                    ver = getattr(result, "version", None) or getattr(result, "model_version", None)
                    logger.info(
                        "MLflow: registered %s version=%s (UI: left sidebar → Models). uri=%s",
                        reg_name,
                        ver,
                        model_uri,
                    )
                    try:
                        mlflow.set_tag(f"registry_{name}", f"{reg_name}:v{ver}" if ver else reg_name)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(
                        "MLflow register_model failed for %s (%s). uri=%s",
                        reg_name,
                        e,
                        model_uri,
                    )
        elif do_register and not run_id:
            logger.warning("MLflow: register_models is true but no active run_id; skipping registry")

    if store:
        logger.info(
            'MLflow: open UI on the SAME store, e.g. mlflow ui --backend-store-uri "%s"',
            store,
        )


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
    fv = effective_feature_version(config)
    mlflow.set_tag("feature_version", fv)
    mlflow.log_param("feature_version", fv)
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
