"""
Configuration utilities
"""
import logging
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)


def infer_project_root(config_path: str) -> Path:
    """
    Repo / project root inferred from the path to a YAML config
    (e.g. ``configs/config.yaml`` → parent of ``configs/``).
    """
    return _project_root_from_config_file(Path(config_path))


def _project_root_from_config_file(config_file: Path) -> Path:
    """
    Infer project root from the resolved config path.
    If the config lives in a ``configs/`` directory, use its parent; otherwise
    use the directory containing the config file.
    """
    resolved = config_file.resolve()
    parent = resolved.parent
    if parent.name == "configs":
        return parent.parent
    return parent


def _resolve_relative_paths(config: Dict[str, Any], project_root: Path) -> None:
    """Turn configured relative paths into absolute paths anchored at project_root."""
    data = config.get("data") or {}
    for key in ("raw_data_path", "processed_data_path"):
        if key not in data or not data[key]:
            continue
        p = Path(data[key])
        if not p.is_absolute():
            data[key] = str((project_root / p).resolve())

    paths = config.get("paths") or {}
    for key in ("models_dir", "logs_dir", "results_dir"):
        if key not in paths or not paths[key]:
            continue
        p = Path(paths[key])
        if not p.is_absolute():
            paths[key] = str((project_root / p).resolve())

    ml = config.get("mlflow")
    if isinstance(ml, dict):
        _resolve_mlflow_file_tracking_uri(ml, project_root)


def _resolve_mlflow_file_tracking_uri(ml: Dict[str, Any], project_root: Path) -> None:
    """
    Anchor ``file:./mlruns`` (and similar relative file URIs) to *project_root*,
    not the process CWD (e.g. Jupyter under ``notebooks/``).
    """
    uri = (ml.get("tracking_uri") or "").strip()
    if not uri.lower().startswith("file:"):
        return
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "file":
        return
    path_str = parsed.path or ""
    if not path_str and parsed.netloc:
        path_str = f"//{parsed.netloc}"
    p = Path(path_str)
    if p.is_absolute():
        return
    resolved = (project_root / p).resolve()
    ml["tracking_uri"] = resolved.as_uri()


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    project_root = _project_root_from_config_file(config_file)
    _resolve_relative_paths(config, project_root)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config
