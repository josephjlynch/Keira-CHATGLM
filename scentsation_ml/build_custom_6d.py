#!/usr/bin/env python3
"""
Build ``datasets/custom_6d.csv`` from raw labeled sessions in ``data/raw_labeled/``.

Each input CSV row is one ~4 Hz sample. Windows are 10 s (40 samples) with 50% overlap
(step 20). Features are computed with :func:`scentsation_hub.compute_features` on each
window — identical to live hub inference.

Run from repo root:
  python -m scentsation_ml.build_custom_6d \\
      --input-dir data/raw_labeled \\
      --output datasets/custom_6d.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scentsation_hub import HUB_FEATURE_NAMES, SensorReading, compute_features

# Expected columns in raw collector CSVs
RAW_COLS = ("time", "subject_id", "label", "session_id", "gsr", "hr", "hrv")


def _parse_time(s: str) -> float:
    s = s.strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float(s)


def _load_session(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Return list of row dicts and optional skip reason."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        missing = set(RAW_COLS) - set(r.fieldnames or [])
        if missing:
            return [], f"missing columns {sorted(missing)}"
        for row in r:
            rows.append(row)
    if len(rows) < 2:
        return [], "fewer than 2 rows"

    s0 = rows[0].get("subject_id", "")
    l0 = rows[0].get("label", "").strip().upper()
    sess0 = rows[0].get("session_id", "")
    for i, row in enumerate(rows):
        if row.get("subject_id", "") != s0 or row.get("session_id", "") != sess0:
            return [], f"inconsistent subject_id/session_id at row {i}"
        if row.get("label", "").strip().upper() != l0:
            return [], f"inconsistent label at row {i}"

    times: list[float] = []
    for row in rows:
        try:
            times.append(_parse_time(str(row["time"])))
        except (ValueError, KeyError):
            times.append(float("nan"))

    if len(times) >= 2 and not all(np.isnan(times)):
        clean = [times[i + 1] - times[i] for i in range(len(times) - 1) if not np.isnan(times[i + 1]) and not np.isnan(times[i])]
        if clean:
            med_ms = 1000.0 * float(np.median(clean))
            if not (200.0 <= med_ms <= 300.0):
                return [], f"median sample interval {med_ms:.0f} ms not in [200, 300] (expected ~4 Hz)"

    return rows, None


def _window_valid_fraction(rows: list[dict[str, Any]], start: int, end: int) -> float:
    """Fraction of samples with HR>=0 and HRV>=0 (PPG not in sentinel state)."""
    chunk = rows[start:end]
    if not chunk:
        return 0.0
    ok = 0
    for row in chunk:
        try:
            hr = float(row["hr"])
            hrv = float(row["hrv"])
        except (KeyError, ValueError, TypeError):
            continue
        if hr >= 0.0 and hrv >= 0.0:
            ok += 1
    return ok / len(chunk)


def _rows_to_readings(rows: list[dict[str, Any]], start: int, end: int) -> deque[SensorReading]:
    d: deque[SensorReading] = deque()
    for row in rows[start:end]:
        gsr = float(row["gsr"])
        hr = float(row["hr"])
        hrv = float(row["hrv"])
        ts = _parse_time(str(row["time"])) if row.get("time") else 0.0
        d.append(SensorReading(gsr=gsr, hr=hr, hrv=hrv, timestamp=ts))
    return d


def build_windows(
    rows: list[dict[str, Any]],
    *,
    window_sec: float,
    overlap: float,
    min_valid_fraction: float,
    label: str,
    subject_id: str,
    session_id: str,
) -> list[dict[str, Any]]:
    sr = 4.0
    win = max(2, int(round(window_sec * sr)))
    step = max(1, int(round(win * (1.0 - overlap))))
    out: list[dict[str, Any]] = []
    n = len(rows)
    widx = 0
    for start in range(0, n - win + 1, step):
        end = start + win
        if _window_valid_fraction(rows, start, end) < min_valid_fraction:
            continue
        feats = compute_features(_rows_to_readings(rows, start, end))
        row_out: dict[str, Any] = {
            "subject_id": subject_id,
            "session_id": session_id,
            "window_idx": widx,
            "label": label,
        }
        for i, name in enumerate(HUB_FEATURE_NAMES):
            row_out[name] = float(feats[i])
        out.append(row_out)
        widx += 1
    return out


def _summarize(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No window rows produced.")
        return

    labels = sorted({r["label"] for r in rows})
    subjects = sorted({r["subject_id"] for r in rows})

    print("\n--- Window counts ---")
    by_label: dict[str, int] = {L: 0 for L in labels}
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
    for L in sorted(by_label):
        n = by_label[L]
        print(f"  {L}: {n}")
        if n < 20:
            print(f"    [warn] fewer than 20 windows for class {L}")

    print("\n--- Per-subject label coverage ---")
    for subj in subjects:
        have = {r["label"] for r in rows if r["subject_id"] == subj}
        missing = [L for L in ("NEUTRAL", "RELAXED", "STRESSED", "FOCUSED") if L not in have]
        if missing:
            print(f"  {subj}: missing classes {missing}")
        else:
            print(f"  {subj}: all 4 classes present")

    feat_cols = list(HUB_FEATURE_NAMES)
    print("\n--- Mean feature vector by class (sanity: STRESSED often higher gsr_mean, lower hrv_rmssd vs RELAXED) ---")
    for L in sorted(by_label):
        sub = [r for r in rows if r["label"] == L]
        print(f"  [{L}] n={len(sub)}")
        for name in feat_cols:
            vals = [r[name] for r in sub]
            m = float(np.mean(vals))
            s = float(np.std(vals))
            print(f"    {name}: mean={m:.4f}  std={s:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build custom_6d.csv from raw labeled CSVs")
    p.add_argument("--input-dir", type=Path, default=_ROOT / "data" / "raw_labeled")
    p.add_argument("--output", type=Path, default=_ROOT / "datasets" / "custom_6d.csv")
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--min-valid-fraction", type=float, default=0.8)
    args = p.parse_args()

    if not 0.0 <= args.overlap < 1.0:
        print("--overlap must be in [0, 1)", file=sys.stderr)
        sys.exit(1)
    if not 0.0 < args.min_valid_fraction <= 1.0:
        print("--min-valid-fraction must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict[str, Any]] = []
    for path in sorted(args.input_dir.glob("*.csv")):
        session_rows, skip = _load_session(path)
        if skip:
            print(f"Skip {path.name}: {skip}")
            continue
        sid = session_rows[0].get("session_id", path.stem)
        subj = session_rows[0].get("subject_id", "unknown")
        lab = session_rows[0].get("label", "NEUTRAL").strip().upper()
        wins = build_windows(
            session_rows,
            window_sec=args.window_sec,
            overlap=args.overlap,
            min_valid_fraction=args.min_valid_fraction,
            label=lab,
            subject_id=str(subj),
            session_id=str(sid),
        )
        print(f"{path.name}: {len(session_rows)} samples → {len(wins)} windows")
        all_rows.extend(wins)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "session_id", "window_idx", "label"] + list(HUB_FEATURE_NAMES)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(f"\nWrote {len(all_rows)} rows → {args.output}")
    _summarize(all_rows)


if __name__ == "__main__":
    main()
