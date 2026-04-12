"""Export trained artifacts.

**15-D pipeline:** Window features from ``features/extractor.py`` — not plug-compatible with the live hub.

**6-D hub contract:** Canonical feature names live in the repo file ``config/hub_feature_names.json``
(same source as ``scentsation_hub``). On-device inference uses streaming features from
``scentsation_hub.compute_features``. The hub's ``hrv_rmssd`` is derived from successive HRV samples
in a rolling buffer — not the same as ECG RR-based RMSSD in seconds from the 15-D extractor.
Train a dedicated 6-D model on features that match the hub definitions, then save with
:func:`build_hub_joblib_payload` / :func:`dump_hub_joblib`.

Do **not** point ``export_for_hub`` at a 15-D ``best_model.joblib`` and expect correct live labels.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import yaml
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_HUB_FEATURE_DIM = 6
_FALLBACK_HUB_FEATURE_NAMES: Tuple[str, ...] = (
    "gsr_mean",
    "gsr_max",
    "gsr_std",
    "gsr_slope",
    "hr_mean",
    "hrv_rmssd",
)


def _hub_feature_names_json_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "hub_feature_names.json"


def load_hub_feature_names_tuple() -> Tuple[str, ...]:
    """Load ``config/hub_feature_names.json``; must match ``scentsation_hub`` (single source of truth)."""
    path = _hub_feature_names_json_path()
    strict = os.getenv("SCENTSATION_REQUIRE_HUB_FEATURE_JSON", "").strip() == "1"
    if not path.is_file():
        if strict:
            raise FileNotFoundError(
                f"Strict mode (SCENTSATION_REQUIRE_HUB_FEATURE_JSON=1): missing {path}"
            )
        logger.warning(
            "config/hub_feature_names.json missing at %s — using fallback names",
            path,
        )
        return _FALLBACK_HUB_FEATURE_NAMES
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != _HUB_FEATURE_DIM:
        raise ValueError(
            f"{path} must be a JSON array of length {_HUB_FEATURE_DIM}, got {data!r}"
        )
    return tuple(str(x) for x in data)


HUB_FEATURE_NAMES: Tuple[str, ...] = load_hub_feature_names_tuple()


def build_hub_joblib_payload(
    model: Any,
    scaler: Any = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a joblib-serializable dict for ``scentsation_hub.load_model_payload`` / ``validate_model_for_hub``.

    Always includes ``feature_names`` so the hub can reject mis-ordered exports when validation runs.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "scaler": scaler,
        "feature_names": list(HUB_FEATURE_NAMES),
    }
    if config is not None:
        payload["config"] = config
    if extra:
        payload.update(extra)
    return payload


def dump_hub_joblib(
    model: Any,
    path: str,
    scaler: Any = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    compress: int = 3,
) -> None:
    """Write a hub-compatible bundle to ``path`` (6-D contract)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = build_hub_joblib_payload(model, scaler, config=config)
    joblib.dump(payload, path, compress=compress)
    logger.info("Wrote hub 6-D bundle to %s", path)


def export_best_model(
    model: Any,
    scaler: StandardScaler,
    model_path: str = "models/best_model.joblib",
    scaler_path: str = "models/scaler.joblib",
    metadata: Optional[Dict] = None,
) -> None:
    """Write model bundle + separate scaler file."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "metadata": metadata or {}},
        model_path,
        compress=3,
    )
    joblib.dump(scaler, scaler_path, compress=3)


def export_for_hub(config_path: str = "config.yaml", model_dir: str = "models") -> None:
    """Mirror ``best_model.joblib`` to ``classifier.joblib`` (still 15-D — not plug-compatible with 6-D ``scentsation_hub``)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    best_path = cfg["output"].get("best_model_path", "models/best_model.joblib")
    scaler_path = cfg["output"].get("scaler_path", "models/scaler.joblib")
    hub_path = os.path.join(model_dir, "classifier.joblib")
    payload = joblib.load(best_path)
    joblib.dump(payload, hub_path, compress=3)
    logger.info("Hub classifier symlink-style export: %s", hub_path)
