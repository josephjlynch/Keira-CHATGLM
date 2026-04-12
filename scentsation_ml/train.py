"""
Scentsation ML — training entry point.

Usage:
  cd scentsation_ml
  pip install -r requirements.txt
  python train.py --config config.yaml --mock-data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.download_wesad import LABEL_TO_INT, generate_mock_wesad
from datasets.download_young_adult import generate_mock_young_adult
from evaluate import generate_report, plot_confusion_matrix, plot_feature_importance, plot_roc_curves
from export import export_for_hub
from features.extractor import extract_features, extract_features_and_labels_from_raw_csv, get_feature_names
from features.windowing import segment_gsr_ecg_pair
from models.knn_classifier import KnnClassifier
from models.mlp_classifier import MlpClassifier
from models.svm_classifier import SvmClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Scentsation.Train")


def load_and_window_dataset(
    processed_dir: str,
    gsr_sr: float,
    ecg_sr: float,
    window_size: float,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CSVs and build (X, y, subject_id)."""
    if not os.path.isdir(processed_dir):
        return np.empty((0, 15)), np.array([]), np.array([])
    files = sorted(
        f for f in os.listdir(processed_dir) if f.endswith(".csv")
    )
    all_X: List[np.ndarray] = []
    all_y: List[int] = []
    all_subj: List[str] = []
    for fn in files:
        path = os.path.join(processed_dir, fn)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.error("Read failed %s: %s", path, e)
            continue
        if "gsr" not in df.columns or "ecg" not in df.columns or "label_int" not in df.columns:
            continue
        subject_id = str(df["subject_id"].iloc[0]) if "subject_id" in df.columns else fn
        gsr_signal = df["gsr"].values.astype(np.float64)
        ecg_signal = df["ecg"].values.astype(np.float64)
        label_ints = df["label_int"].values.astype(int)
        gsr_w, ecg_w, win_lab = segment_gsr_ecg_pair(
            gsr_signal, ecg_signal, gsr_sr, ecg_sr, label_ints, window_size, overlap
        )
        mask = win_lab >= 0
        for gw, ew, lbl in zip(
            [g for g, m in zip(gsr_w, mask) if m],
            [e for e, m in zip(ecg_w, mask) if m],
            win_lab[mask],
        ):
            try:
                feats = extract_features(gw, ew, gsr_sr, ecg_sr)
            except Exception as e:
                logger.warning("Feature error %s: %s", subject_id, e)
                continue
            all_X.append(feats)
            all_y.append(int(lbl))
            all_subj.append(subject_id)
    if not all_X:
        return np.empty((0, 15)), np.array([]), np.array([])
    return np.asarray(all_X, dtype=np.float64), np.asarray(all_y), np.asarray(all_subj)


def subject_wise_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Leave entire subjects out of splits."""
    rng = np.random.RandomState(random_state)
    u = np.unique(subjects)
    rng.shuffle(u)
    n = len(u)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    tr_s, va_s, te_s = u[:n_train], u[n_train : n_train + n_val], u[n_train + n_val :]
    if len(te_s) == 0:
        te_s = va_s[-1:]
        va_s = va_s[:-1]
    tr_m = np.isin(subjects, tr_s)
    va_m = np.isin(subjects, va_s)
    te_m = np.isin(subjects, te_s)
    return {
        "X_train": X[tr_m],
        "y_train": y[tr_m],
        "X_val": X[va_m],
        "y_val": y[va_m],
        "X_test": X[te_m],
        "y_test": y[te_m],
    }


def generate_full_mock_data(config: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Populate mock CSVs and load windowed features."""
    wcfg = config["window"]
    gsr_sr = float(wcfg["gsr_sample_rate"])
    ecg_sr = float(wcfg["ecg_sample_rate"])
    win_size = float(wcfg["size_sec"])
    overlap = float(wcfg["overlap_ratio"])
    wdir = config["data"]["wesad"]["processed_dir"]
    ydir = config["data"]["young_adult"]["processed_dir"]
    generate_mock_wesad(
        n_subjects=12,
        duration_sec_per_condition=90,
        output_dir=wdir,
        gsr_sr=gsr_sr,
        ecg_sr=ecg_sr,
    )
    generate_mock_young_adult(
        n_subjects=6,
        duration_sec_per_condition=60,
        output_dir=ydir,
        gsr_sr=gsr_sr,
        ecg_sr=ecg_sr,
    )
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    Xw, yw, sw = load_and_window_dataset(wdir, gsr_sr, ecg_sr, win_size, overlap)
    if len(Xw):
        out["WESAD"] = (Xw, yw, sw)
    Xy, yy, sy = load_and_window_dataset(ydir, gsr_sr, ecg_sr, win_size, overlap)
    if len(Xy):
        out["YoungAdult"] = (Xy, yy, sy)
    return out


def train_single_model(
    model_name: str,
    model_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: List[str],
) -> Tuple[str, object, Dict]:
    """Train one model and evaluate on validation."""
    model_config = {**model_config, "class_names": class_names}
    if model_name == "SVM":
        clf = SvmClassifier(model_config)
    elif model_name == "MLP":
        clf = MlpClassifier(model_config)
    elif model_name == "KNN":
        clf = KnnClassifier(model_config)
    else:
        raise ValueError(model_name)
    t0 = time.time()
    tm = clf.train(X_train, y_train, X_val, y_val)
    logger.info("[%s] train %.1fs", model_name, time.time() - t0)
    vm = clf.evaluate(X_val, y_val)
    return model_name, clf, {**vm, **tm, "train_time_sec": time.time() - t0}


def run_pipeline(
    config: dict,
    mock_data: bool = False,
    finetune_mode: bool = False,
    custom_data_path: Optional[str] = None,
) -> None:
    """End-to-end train / optional finetune."""
    wcfg = config["window"]
    gsr_sr = float(wcfg["gsr_sample_rate"])
    ecg_sr = float(wcfg["ecg_sample_rate"])
    win_size = float(wcfg["size_sec"])
    overlap = float(wcfg["overlap_ratio"])
    class_names = config["classes"]["names"]
    out_cfg = config["output"]
    os.makedirs(out_cfg["model_dir"], exist_ok=True)
    os.makedirs(out_cfg["results_dir"], exist_ok=True)

    if finetune_mode and custom_data_path:
        _run_finetune(config, custom_data_path, class_names, out_cfg)
        return

    if mock_data:
        datasets = generate_full_mock_data(config)
    else:
        datasets = {}
        wd = config["data"]["wesad"]["processed_dir"]
        if os.path.isdir(wd) and os.listdir(wd):
            X, y, s = load_and_window_dataset(wd, gsr_sr, ecg_sr, win_size, overlap)
            if len(X):
                datasets["WESAD"] = (X, y, s)
        yd = config["data"]["young_adult"]["processed_dir"]
        if os.path.isdir(yd) and os.listdir(yd):
            X, y, s = load_and_window_dataset(yd, gsr_sr, ecg_sr, win_size, overlap)
            if len(X):
                datasets["YA"] = (X, y, s)

    if not datasets:
        logger.error("No data. Use --mock-data or add processed CSVs.")
        return

    X_list = [v[0] for v in datasets.values()]
    y_list = [v[1] for v in datasets.values()]
    s_list = [v[2] for v in datasets.values()]
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    subj_all = np.concatenate(s_list)
    logger.info("Samples=%d subjects=%d", len(X_all), len(np.unique(subj_all)))

    sp = config["training"]["subject_split"]
    splits = subject_wise_split(
        X_all,
        y_all,
        subj_all,
        train_ratio=float(sp["train_ratio"]),
        val_ratio=float(sp["val_ratio"]),
        random_state=int(config["training"]["random_state"]),
    )

    model_cfgs = {
        "SVM": config["models"]["svm"],
        "MLP": config["models"]["mlp"],
        "KNN": config["models"]["knn"],
    }
    # Sequential training avoids joblib/loky edge cases on some Python builds.
    results_list = [
        train_single_model(
            name,
            model_cfgs[name],
            splits["X_train"],
            splits["y_train"],
            splits["X_val"],
            splits["y_val"],
            class_names,
        )
        for name in model_cfgs
    ]

    trained = {}
    val_results: Dict[str, Dict] = {}
    all_results: Dict[str, Dict] = {}
    for name, clf, metrics in results_list:
        trained[name] = clf
        val_results[name] = {"f1_macro": metrics["f1_macro"], **metrics}
        all_results[name] = metrics
        logger.info("[%s] val F1=%.4f", name, metrics["f1_macro"])

    best_name = max(val_results, key=lambda k: val_results[k]["f1_macro"])
    best = trained[best_name]
    logger.info("Best model: %s", best_name)

    test_eval = best.evaluate(splits["X_test"], splits["y_test"])
    all_results[best_name].update(test_eval)
    feature_names = get_feature_names()

    plot_confusion_matrix(
        test_eval["confusion_matrix"],
        class_names,
        out_cfg["confusion_matrix_path"],
    )
    try:
        proba = best.predict_proba(splits["X_test"])
        plot_roc_curves(
            splits["y_test"],
            proba,
            class_names,
            out_cfg["roc_curve_path"],
        )
    except Exception as e:
        logger.warning("ROC skipped: %s", e)

    imp = best.get_feature_importance(splits["X_test"], splits["y_test"])
    if imp is not None and len(imp) == len(feature_names):
        plot_feature_importance(imp, feature_names, out_cfg["feature_importance_path"])

    generate_report(all_results, feature_names, out_cfg["report_path"])
    best.save(out_cfg["best_model_path"])
    export_for_hub("config.yaml", out_cfg["model_dir"])
    logger.info("Saved %s", out_cfg["best_model_path"])


def _run_finetune(
    config: dict,
    custom_data_path: str,
    class_names: List[str],
    out_cfg: dict,
) -> None:
    """Fine-tune best-effort on custom tabular CSV."""
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier as SkMLP

    best_path = out_cfg["best_model_path"]
    if not os.path.isfile(best_path):
        logger.error("Missing %s — train without --finetune first.", best_path)
        return
    df = pd.read_csv(custom_data_path)
    if "label_int" not in df.columns and "label" in df.columns:
        df["label_int"] = df["label"].map(lambda x: LABEL_TO_INT.get(str(x).strip(), 0))
    X, y = extract_features_and_labels_from_raw_csv(df)
    if len(X) < 2:
        logger.error("Not enough windows from custom data.")
        return
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=int(config["training"]["random_state"])
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=int(config["training"]["random_state"])
    )
    ft = config["training"]["finetune"]
    payload = joblib.load(best_path)
    pre = payload["model"]
    mlp_cfg = config["models"]["mlp"]
    mlp_cfg["class_names"] = class_names
    mlp_cfg["finetune_lr"] = float(ft["learning_rate"])
    mlp_cfg["finetune_epochs"] = int(ft["additional_epochs"])
    clf = MlpClassifier(mlp_cfg)
    if isinstance(pre, SkMLP):
        clf.model = pre
        clf.scaler = payload.get("scaler")
        clf.is_trained = True
        clf.finetune(X_train, y_train, X_val, y_val)
    else:
        logger.warning("Best checkpoint is %s — training a fresh MLP on custom data.", type(pre))
        clf.train(X_train, y_train, X_val, y_val)
    ev = clf.evaluate(X_test, y_test)
    logger.info("Finetune test F1=%.4f", ev["f1_macro"])
    clf.save(out_cfg["finetuned_model_path"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mock-data", action="store_true")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--custom-data", default=None)
    args = ap.parse_args()
    if not os.path.isfile(args.config):
        logger.error("Config not found: %s", args.config)
        sys.exit(1)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_pipeline(
        cfg,
        mock_data=args.mock_data,
        finetune_mode=args.finetune,
        custom_data_path=args.custom_data,
    )


if __name__ == "__main__":
    main()
