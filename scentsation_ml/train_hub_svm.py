#!/usr/bin/env python3
"""
Train a 6-D RBF SVM (StandardScaler + SVC) for :mod:`scentsation_hub` inference.

Loads windowed features from ``datasets/custom_6d.csv`` (see ``build_custom_6d``),
uses subject-wise outer holdout and grouped inner CV, then exports a hub-compatible
joblib via :func:`export.dump_hub_joblib`.

Phase 3 — Hub smoke test (from repo root)
------------------------------------------

With **only one Arduino** (sensors) and no second USB device for pumps, use mock
sensors so the hub skips opening both serial ports::

    python scentsation_hub.py --model-path models/classifier.joblib --mock-mode --no-llm

Non-mock mode runs ``preflight_serial_ports`` for **both** ``--esp32-port`` and
``--arduino-port``; sensor-only setups should use ``--mock-mode`` for this check
or provide two valid ports.

Phase 4 — Anti-memorization (subject holdout)
----------------------------------------------

The headline metric is **test macro-F1 on held-out subject_ids** (outer
``GroupShuffleSplit``), not accuracy on a single subject. Inner ``GroupKFold``
macro-F1 is only for hyperparameter search on **training** subjects.

Use ``--outer-repeats N`` (``N>1``) to print mean/std/min/max of test macro-F1
across ``random_state``, ``random_state+1``, … splits; the exported model still
uses the primary split (``--random-state``) unless you pass ``--refit-on-all``.

If FOCUSED ↔ STRESSED confusions are high, try ``--class-weight-json`` (per-class
weights) or duplicate underrepresented windows in the CSV. Warnings use
``--warn-focused-stressed-ratio`` (fraction of true-class support on the test set).

Phase 5 — Venue calibration (demo machine)
-------------------------------------------

Before the conference, record 2–3 short labeled sessions on the **booth** laptop
(same HVAC/lighting as the demo) with a distinct ``subject_id`` (e.g.
``venue_cal_2026_04_17``), save CSVs under ``data/raw_labeled/``, then rebuild
windows and retrain::

    python -m scentsation_ml.build_custom_6d \\
        --input-dir data/raw_labeled --output datasets/custom_6d.csv
    python scentsation_ml/train_hub_svm.py --data datasets/custom_6d.csv \\
        --results-dir results/venue

Or run ``scripts/venue_retrain.sh`` from the repo root.

Requires: pandas, scikit-learn, joblib; optional matplotlib for confusion-matrix PNG.

Training defaults to ``--n-jobs 1`` for GridSearchCV (reliable on Python 3.14 / joblib); pass ``--n-jobs -1`` to use all CPU cores where loky works.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scentsation_hub import CLASS_ORDER  # noqa: E402
from scentsation_ml.export import HUB_FEATURE_NAMES, dump_hub_joblib  # noqa: E402

logger = logging.getLogger(__name__)

LABEL_TO_INT = {name: i for i, name in enumerate(CLASS_ORDER)}


def _resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def load_xy_groups(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load features X, labels y (int), groups (subject_id strings), and raw frame."""
    df = pd.read_csv(csv_path)
    need = {"subject_id", "label", *HUB_FEATURE_NAMES}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(need)).copy()
    for c in HUB_FEATURE_NAMES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(HUB_FEATURE_NAMES))

    labels_upper = df["label"].astype(str).str.strip().str.upper()
    unknown = sorted(set(labels_upper) - set(CLASS_ORDER))
    if unknown:
        raise ValueError(f"Unknown labels (expected one of {CLASS_ORDER}): {unknown}")
    y = labels_upper.map(LABEL_TO_INT).astype(np.int64).values
    X = df[list(HUB_FEATURE_NAMES)].values.astype(np.float64)
    groups = df["subject_id"].astype(str).values
    return X, y, groups, df


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, float) and not np.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


def _load_class_weight_json(path: Path) -> Dict[int, float]:
    """Load JSON mapping class name (or int index) to positive weight for SVC."""
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"class_weight JSON must be a non-empty object: {path}")
    out: Dict[int, float] = {}
    for k, v in raw.items():
        if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
            idx = int(k)
        else:
            ku = str(k).strip().upper()
            if ku not in LABEL_TO_INT:
                raise ValueError(f"Unknown class key {k!r} in {path}")
            idx = LABEL_TO_INT[ku]
        out[idx] = float(v)
    for i in range(len(CLASS_ORDER)):
        out.setdefault(i, 1.0)
    return out


def _parse_class_weight_arg(arg: str) -> Union[str, Dict[int, float]]:
    p = Path(arg)
    candidates = [p, _REPO_ROOT / arg]
    for c in candidates:
        if c.is_file():
            return _load_class_weight_json(c.resolve())
    raise FileNotFoundError(f"class_weight JSON not found: {arg!r} (tried {candidates})")


def _train_support_str(y_train: np.ndarray) -> str:
    parts = []
    for i, name in enumerate(CLASS_ORDER):
        parts.append(f"{name}={int(np.sum(y_train == i))}")
    return "train_per_class_support: " + ", ".join(parts)


def _warn_focused_stressed(
    y_test: np.ndarray,
    cm: np.ndarray,
    ratio: float,
) -> None:
    i_f = LABEL_TO_INT["FOCUSED"]
    i_s = LABEL_TO_INT["STRESSED"]
    n_f = int(np.sum(y_test == i_f))
    n_s = int(np.sum(y_test == i_s))
    err_fs = int(cm[i_f, i_s])
    err_sf = int(cm[i_s, i_f])
    thr_f = max(1.0, ratio * n_f) if n_f else float("inf")
    thr_s = max(1.0, ratio * n_s) if n_s else float("inf")
    if n_f and err_fs > thr_f:
        logger.warning(
            "FOCUSED→STRESSED confusion %d exceeds ratio*support (%.2f * %d = %.1f). "
            "Consider --class-weight-json or oversampling.",
            err_fs,
            ratio,
            n_f,
            thr_f,
        )
    if n_s and err_sf > thr_s:
        logger.warning(
            "STRESSED→FOCUSED confusion %d exceeds ratio*support (%.2f * %d = %.1f). "
            "Consider --class-weight-json or oversampling.",
            err_sf,
            ratio,
            n_s,
            thr_s,
        )


@dataclass
class OuterFoldResult:
    search: GridSearchCV
    macro_f1: float
    y_test: np.ndarray
    y_pred: np.ndarray
    test_subject_ids: List[str]
    train_subject_ids: List[str]
    train_idx: np.ndarray
    test_idx: np.ndarray


def _fit_one_outer_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_subject_fraction: float,
    outer_seed: int,
    cv_splits: int,
    n_jobs: int,
    random_state: int,
    class_weight: Union[str, Dict[int, float]],
    param_grid: Dict[str, List[Any]],
) -> Optional[OuterFoldResult]:
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=test_subject_fraction,
        random_state=outer_seed,
    )
    train_idx, test_idx = next(outer.split(X, y, groups=groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    groups_train = groups[train_idx]
    test_subject_ids = sorted(set(groups[test_idx]))
    train_subject_ids = sorted(set(groups_train))
    n_train_subj = len(train_subject_ids)

    if n_train_subj < 2:
        logger.warning(
            "Skipping outer seed %s: only %d training subject(s) (need >=2 for GroupKFold).",
            outer_seed,
            n_train_subj,
        )
        return None

    n_inner = min(cv_splits, n_train_subj)
    if n_inner < 2:
        n_inner = 2
    inner_cv = GroupKFold(n_splits=n_inner)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight=class_weight,
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups_train)
    best = search.best_estimator_
    y_pred = best.predict(X_test)
    labels_idx = list(range(len(CLASS_ORDER)))
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels_idx, zero_division=0)
    logger.info(
        "Outer seed %s: test_macro_f1=%.4f  test_subjects=%s",
        outer_seed,
        macro_f1,
        test_subject_ids,
    )
    return OuterFoldResult(
        search=search,
        macro_f1=macro_f1,
        y_test=y_test,
        y_pred=y_pred,
        test_subject_ids=test_subject_ids,
        train_subject_ids=train_subject_ids,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(description="Train 6-D hub RBF SVM from custom_6d.csv")
    p.add_argument("--data", type=Path, default=Path("datasets/custom_6d.csv"))
    p.add_argument("--output-model", type=Path, default=Path("models/classifier.joblib"))
    p.add_argument("--test-subject-fraction", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--outer-repeats",
        type=int,
        default=1,
        help="Number of outer GroupShuffleSplit draws with seeds random_state+k (k=0..N-1). "
        "N>1 prints stability of test macro-F1; export uses k=0 unless --refit-on-all.",
    )
    p.add_argument(
        "--refit-on-all",
        action="store_true",
        help="After holdout evaluation, refit GridSearch on all rows (grouped inner CV) "
        "and export that model (venue-style full-data fit).",
    )
    p.add_argument("--cv-splits", type=int, default=5, help="Max GroupKFold splits (capped by train subjects)")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="GridSearchCV parallel jobs (default 1 avoids joblib/loky issues on some Python 3.14 setups; use -1 for all cores)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="If set, write confusion_matrix_hub_svm.png and metrics_hub_svm.txt here",
    )
    p.add_argument(
        "--class-weight-json",
        type=str,
        default=None,
        help="Path to JSON object mapping class name (or 0..3 index) to weight; replaces SVC class_weight='balanced'.",
    )
    p.add_argument(
        "--warn-focused-stressed-ratio",
        type=float,
        default=0.15,
        help="Log warning if FOCUSED↔STRESSED off-diagonal count exceeds this fraction of the true-class test count.",
    )
    args = p.parse_args()

    data_path = _resolve_path(args.data)
    out_path = _resolve_path(args.output_model)
    if not data_path.is_file():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    X, y, groups, df = load_xy_groups(data_path)
    n = len(df)
    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)
    logger.info("Loaded %d rows, %d subjects, %d features", n, n_subjects, X.shape[1])

    if n_subjects < 2:
        logger.error("Need at least 2 distinct subject_id values for a grouped holdout.")
        sys.exit(1)

    class_weight: Union[str, Dict[int, float]] = (
        _parse_class_weight_arg(args.class_weight_json) if args.class_weight_json else "balanced"
    )
    if isinstance(class_weight, dict):
        logger.info("Using custom SVC class_weight=%s", class_weight)

    param_grid = {
        "svm__C": [0.1, 1, 3, 10, 30, 100],
        "svm__gamma": ["scale", 0.01, 0.03, 0.1, 0.3, 1.0],
    }

    repeat_scores: List[Tuple[int, float, List[str]]] = []
    primary: Optional[OuterFoldResult] = None

    for k in range(max(1, args.outer_repeats)):
        outer_seed = args.random_state + k
        res = _fit_one_outer_split(
            X,
            y,
            groups,
            args.test_subject_fraction,
            outer_seed,
            args.cv_splits,
            args.n_jobs,
            args.random_state,
            class_weight,
            param_grid,
        )
        if res is None:
            if k == 0:
                logger.error(
                    "Primary outer split (seed %s) invalid: not enough training subjects for GroupKFold. "
                    "Lower --test-subject-fraction or add subjects.",
                    outer_seed,
                )
                sys.exit(1)
            continue
        repeat_scores.append((outer_seed, res.macro_f1, res.test_subject_ids))
        if k == 0:
            primary = res

    if primary is None:
        logger.error("No valid outer split (check --test-subject-fraction and subject count).")
        sys.exit(1)

    if len(repeat_scores) > 1:
        scores = np.array([s[1] for s in repeat_scores], dtype=np.float64)
        print("\n--- Outer repeat stability (test macro-F1, held-out subjects) ---")
        for seed, f1, subs in repeat_scores:
            print(f"  seed={seed}  macro_f1={f1:.4f}  test_subjects={subs}")
        print(
            f"  aggregate: mean={scores.mean():.4f}  std={scores.std(ddof=1) if len(scores) > 1 else 0.0:.4f}  "
            f"min={scores.min():.4f}  max={scores.max():.4f}"
        )

    search = primary.search
    y_test, y_pred = primary.y_test, primary.y_pred
    test_subject_ids = primary.test_subject_ids
    train_subject_ids = primary.train_subject_ids
    y_train = y[primary.train_idx]

    labels_idx = list(range(len(CLASS_ORDER)))
    report = classification_report(
        y_test,
        y_pred,
        labels=labels_idx,
        target_names=list(CLASS_ORDER),
        zero_division=0,
    )
    print("\n--- Held-out subjects (primary outer split, k=0) ---")
    print(f"Test subjects ({len(test_subject_ids)}): {test_subject_ids}")
    print("\n--- Classification report (test) ---\n", report)

    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels_idx, zero_division=0)
    print(f"Test macro-F1 (primary): {macro_f1:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
    print("\n--- Confusion matrix (rows=true, cols=pred) ---")
    header = "          " + "  ".join(f"{c:>10}" for c in CLASS_ORDER)
    print(header)
    for i, row_name in enumerate(CLASS_ORDER):
        row = "  ".join(f"{cm[i, j]:>10}" for j in range(len(CLASS_ORDER)))
        print(f"{row_name:>10}  {row}")

    i_focused = LABEL_TO_INT["FOCUSED"]
    i_stressed = LABEL_TO_INT["STRESSED"]
    print("\n--- Demo-critical confusions (FOCUSED vs STRESSED) ---")
    n_f = int(np.sum(y_test == i_focused))
    n_s = int(np.sum(y_test == i_stressed))
    c_fs = int(cm[i_focused, i_stressed])
    c_sf = int(cm[i_stressed, i_focused])
    print(
        f"  True FOCUSED, pred STRESSED:  cm[{i_focused},{i_stressed}] = {c_fs}"
        + (f"  (rate vs true FOCUSED test n={n_f}: {c_fs / max(n_f, 1):.3f})" if n_f else "")
    )
    print(
        f"  True STRESSED, pred FOCUSED: cm[{i_stressed},{i_focused}] = {c_sf}"
        + (f"  (rate vs true STRESSED test n={n_s}: {c_sf / max(n_s, 1):.3f})" if n_s else "")
    )

    _warn_focused_stressed(y_test, cm, args.warn_focused_stressed_ratio)

    export_model = search.best_estimator_
    export_search = search
    refit_note = ""

    if args.refit_on_all:
        n_inner_full = min(args.cv_splits, n_subjects)
        if n_inner_full < 2:
            n_inner_full = 2
        inner_all = GroupKFold(n_splits=n_inner_full)
        pipe_all = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        class_weight=class_weight,
                        probability=True,
                        random_state=args.random_state,
                    ),
                ),
            ],
        )
        search_all = GridSearchCV(
            pipe_all,
            param_grid,
            cv=inner_all,
            scoring="f1_macro",
            n_jobs=args.n_jobs,
            refit=True,
            verbose=1,
        )
        logger.info("Refitting on all subjects (--refit-on-all)")
        search_all.fit(X, y, groups=groups)
        export_model = search_all.best_estimator_
        export_search = search_all
        refit_note = "refit_on_all=true; model fit on all subjects with grouped inner CV.\n"

    if args.results_dir is not None:
        res_dir = _resolve_path(args.results_dir)
        res_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = res_dir / "metrics_hub_svm.txt"
        support_line = _train_support_str(y_train) + "\n"
        metrics_path.write_text(
            refit_note
            + support_line
            + f"best_params={json.dumps(_serialize_params(dict(export_search.best_params_)))}\n"
            f"cv_best_f1_macro={export_search.best_score_:.6f}\n"
            f"test_macro_f1_primary_holdout={macro_f1:.6f}\n"
            f"test_subjects_primary={test_subject_ids}\n"
            + (
                f"outer_repeat_test_macro_f1={json.dumps([s[1] for s in repeat_scores])}\n"
                if len(repeat_scores) > 1
                else ""
            )
            + "\n"
            + report
            + "\n\n"
            + str(cm),
            encoding="utf-8",
        )
        logger.info("Wrote %s", metrics_path)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=list(CLASS_ORDER),
                yticklabels=list(CLASS_ORDER),
                ylabel="True label",
                xlabel="Predicted label",
                title="Hub SVM (held-out subjects, primary split)",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            fig.tight_layout()
            png_path = res_dir / "confusion_matrix_hub_svm.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Wrote %s", png_path)
        except Exception as e:
            logger.warning("Could not save confusion matrix PNG: %s", e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        logger.warning("Overwriting existing model: %s", out_path)

    meta: Dict[str, Any] = {
        "trainer": "train_hub_svm",
        "best_params": _serialize_params(dict(export_search.best_params_)),
        "cv_best_f1_macro": float(export_search.best_score_),
        "test_macro_f1": float(macro_f1),
        "test_subjects": test_subject_ids,
        "train_subjects": train_subject_ids,
        "classes": list(CLASS_ORDER),
        "refit_on_all": bool(args.refit_on_all),
    }
    if len(repeat_scores) > 1:
        meta["outer_repeat_test_macro_f1"] = [float(s[1]) for s in repeat_scores]

    dump_hub_joblib(export_model, str(out_path), scaler=None, config=meta)
    logger.info("Hub bundle written to %s", out_path)
    print(
        f"\nNext: python scentsation_hub.py --model-path {out_path} --mock-mode --no-llm\n"
        "(Use --mock-mode if you do not have two USB serial devices for sensor + pumps.)"
    )


if __name__ == "__main__":
    main()
