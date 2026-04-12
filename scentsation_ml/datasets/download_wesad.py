"""
WESAD preprocessing — place subject ``.pkl`` files under ``data/wesad/raw`` after obtaining access.
See: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/wesad/
"""

import logging
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

RESPIBAN_EDA = 5
RESPIBAN_ECG = 1

LABEL_TO_INT = {"NEUTRAL": 0, "RELAXED": 1, "STRESSED": 2, "FOCUSED": 3}

WESAD_LABELS = {
    0: "NEUTRAL",
    1: "NEUTRAL",
    2: "STRESSED",
    3: "STRESSED",
    4: "RELAXED",
    5: "FOCUSED",
    6: "RELAXED",
    7: "STRESSED",
}


def _resample_signal(sig: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    if orig_sr == target_sr:
        return sig
    n_samples = int(len(sig) * target_sr / orig_sr)
    return scipy_signal.resample(sig, n_samples)


def preprocess_single_subject(
    pkl_path: str, out_dir: str, gsr_sr: float = 4.0, ecg_sr: float = 64.0
) -> Optional[str]:
    """Convert one WESAD ``.pkl`` to a per-sample CSV (``gsr``, ``ecg``, ``label_int``)."""
    subject_id = os.path.basename(pkl_path).replace(".pkl", "")
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        logger.error("Failed to load %s: %s", pkl_path, e)
        return None

    if "signal" not in data or "chest" not in data["signal"]:
        logger.error("Unexpected structure in %s", pkl_path)
        return None

    chest_signal = data["signal"]["chest"]
    raw_lab = data["label"]
    if isinstance(raw_lab, dict) and "chest" in raw_lab:
        chest_label = np.asarray(raw_lab["chest"]).astype(int)
    else:
        chest_label = np.asarray(raw_lab).astype(int)
    orig_sr = 700.0
    eda_raw = chest_signal[RESPIBAN_EDA].astype(np.float64)
    ecg_raw = chest_signal[RESPIBAN_ECG].astype(np.float64)

    gsr = _resample_signal(eda_raw, orig_sr, gsr_sr)
    ecg = _resample_signal(ecg_raw, orig_sr, ecg_sr)
    labels = _resample_signal(chest_label.astype(np.float64), orig_sr, ecg_sr)
    labels = np.round(labels).astype(int)

    label_names = np.array([WESAD_LABELS.get(int(l), "NEUTRAL") for l in labels])
    label_ints = np.array([LABEL_TO_INT.get(ln, 0) for ln in label_names])

    min_len = min(len(gsr), len(ecg), len(label_names))
    df = pd.DataFrame(
        {
            "subject_id": subject_id,
            "sample_idx": np.arange(min_len),
            "gsr": gsr[:min_len],
            "ecg": ecg[:min_len],
            "label": label_names[:min_len],
            "label_int": label_ints[:min_len],
        }
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject_id}.csv")
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return out_path


def preprocess_wesad(
    raw_dir: str, out_dir: str, gsr_sr: float = 4.0, ecg_sr: float = 64.0
) -> List[str]:
    """Process all ``*.pkl`` in ``raw_dir``."""
    if not os.path.isdir(raw_dir):
        logger.error("Missing raw dir: %s", raw_dir)
        return []
    paths = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".pkl"):
            continue
        p = preprocess_single_subject(os.path.join(raw_dir, name), out_dir, gsr_sr, ecg_sr)
        if p:
            paths.append(p)
    return paths


def merge_custom_data(merged_df: pd.DataFrame, custom_path: str, label_col: str = "label") -> pd.DataFrame:
    """Append custom CSV rows to a merged frame."""
    if not os.path.isfile(custom_path):
        return merged_df
    c = pd.read_csv(custom_path)
    if label_col in c.columns:
        c["label_int"] = c[label_col].map(lambda x: LABEL_TO_INT.get(str(x).strip(), 0))
        c = c.dropna(subset=["label_int"])
    return pd.concat([merged_df, c], ignore_index=True)


def generate_mock_wesad(
    n_subjects: int = 6,
    duration_sec_per_condition: int = 60,
    gsr_sr: float = 4.0,
    ecg_sr: float = 64.0,
    output_dir: str = "data/wesad/processed",
) -> List[str]:
    """Synthetic CSVs for ``--mock-data``."""
    os.makedirs(output_dir, exist_ok=True)
    out = []
    classes = list(LABEL_TO_INT.keys())
    for s in range(1, n_subjects + 1):
        rows = []
        for cond in classes:
            n_gsr = int(duration_sec_per_condition * gsr_sr)
            n_ecg = int(duration_sec_per_condition * ecg_sr)
            gsr = np.clip(np.random.normal(3.0, 0.5, n_gsr), 0.1, 15.0)
            t = np.linspace(0, duration_sec_per_condition, n_ecg)
            hr = 70 + LABEL_TO_INT[cond] * 3
            period = 60.0 / hr
            ecg = 0.7 * np.sin(2 * np.pi * t / period) + np.random.normal(0, 0.05, len(t))
            m = min(len(gsr), len(ecg))
            for i in range(m):
                rows.append(
                    {
                        "subject_id": f"MOCK_S{s}",
                        "sample_idx": i,
                        "gsr": float(gsr[i]),
                        "ecg": float(ecg[i]),
                        "label": cond,
                        "label_int": LABEL_TO_INT[cond],
                    }
                )
        path = os.path.join(output_dir, f"MOCK_S{s}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        out.append(path)
    return out
