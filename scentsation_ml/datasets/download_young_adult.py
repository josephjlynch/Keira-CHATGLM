"""Young Adult affective data — place vendor CSVs under ``data/young_adult/raw``."""

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from .download_wesad import LABEL_TO_INT

logger = logging.getLogger(__name__)


def _resample(sig: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    if orig_sr == target_sr:
        return sig
    return scipy_signal.resample(sig, int(len(sig) * target_sr / orig_sr))


def _map_va(v: float, a: float) -> str:
    if v >= 5.0 and a >= 5.0:
        return "FOCUSED"
    if v >= 5.0 and a < 5.0:
        return "RELAXED"
    if v < 5.0 and a >= 5.0:
        return "STRESSED"
    return "NEUTRAL"


def preprocess_young_adult(
    raw_dir: str, out_dir: str, gsr_sr: float = 4.0, ecg_sr: float = 64.0
) -> List[str]:
    """Process CSV files with ``gsr`` + ``ecg`` (+ optional ``valence``/``arousal``)."""
    if not os.path.isdir(raw_dir):
        logger.error("Missing raw dir: %s", raw_dir)
        return []
    results = []
    cols_lower = {}
    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith(".csv"):
            continue
        path = os.path.join(raw_dir, f)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.error("Read failed %s: %s", path, e)
            continue
        cols_lower = {c.lower(): c for c in df.columns}
        if "gsr" not in cols_lower or "ecg" not in cols_lower:
            logger.warning("Skip %s (need gsr+ecg columns)", f)
            continue
        gcol, ecol = cols_lower["gsr"], cols_lower["ecg"]
        gsr = df[gcol].values.astype(np.float64)
        ecg = df[ecol].values.astype(np.float64)
        if "valence" in df.columns and "arousal" in df.columns:
            labels = [_map_va(float(r["valence"]), float(r["arousal"])) for _, r in df.iterrows()]
        elif "label" in df.columns:
            labels = [str(x).strip() for x in df["label"].values]
        else:
            labels = ["NEUTRAL"] * len(df)
        li = [LABEL_TO_INT.get(l, 0) for l in labels]
        m = min(len(gsr), len(ecg), len(labels))
        out = pd.DataFrame(
            {
                "subject_id": f.replace(".csv", ""),
                "sample_idx": np.arange(m),
                "gsr": gsr[:m],
                "ecg": ecg[:m],
                "label": labels[:m],
                "label_int": li[:m],
            }
        )
        os.makedirs(out_dir, exist_ok=True)
        op = os.path.join(out_dir, f)
        out.to_csv(op, index=False)
        results.append(op)
    return results


def generate_mock_young_adult(
    n_subjects: int = 4,
    duration_sec_per_condition: int = 50,
    gsr_sr: float = 4.0,
    ecg_sr: float = 64.0,
    output_dir: str = "data/young_adult/processed",
) -> List[str]:
    """Synthetic Young-Adult-style CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    out = []
    for lab in LABEL_TO_INT:
        for s in range(1, n_subjects + 1):
            n_g = int(duration_sec_per_condition * gsr_sr)
            n_e = int(duration_sec_per_condition * ecg_sr)
            gsr = np.clip(np.random.normal(3.0, 0.4, n_g), 0.1, 15.0)
            t = np.linspace(0, duration_sec_per_condition, n_e)
            period = 60.0 / (68 + LABEL_TO_INT[lab])
            ecg = 0.6 * np.sin(2 * np.pi * t / period) + np.random.normal(0, 0.05, len(t))
            m = min(len(gsr), len(ecg))
            sid = f"MOCK_YA_{lab}_{s}"
            rows = [
                {
                    "subject_id": sid,
                    "sample_idx": i,
                    "gsr": float(gsr[i]),
                    "ecg": float(ecg[i]),
                    "label": lab,
                    "label_int": LABEL_TO_INT[lab],
                }
                for i in range(m)
            ]
            path = os.path.join(output_dir, f"MOCK_YA_{lab}_{s}.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            out.append(path)
    return out
