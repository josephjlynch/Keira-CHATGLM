"""Feature extraction from GSR and ECG windows (15-D vector)."""

import logging
from typing import List

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import linregress

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    "gsr_mean",
    "gsr_std",
    "gsr_max",
    "gsr_min",
    "gsr_slope",
    "gsr_num_peaks",
    "gsr_peak_amp_mean",
    "gsr_first_diff_mean",
    "gsr_first_diff_std",
    "hrv_mean_hr",
    "hrv_std_hr",
    "hrv_rmssd",
    "hrv_sdnn",
    "hrv_pnn50",
    "hrv_lf_hf_ratio",
]


def get_feature_names() -> List[str]:
    """Return ordered feature column names."""
    return list(FEATURE_NAMES)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return float(fn(y, x))
    return float(np.trapz(y, x))


def _gsr_time_axis(window: np.ndarray, sample_rate: float) -> np.ndarray:
    return np.arange(len(window)) / sample_rate


def extract_gsr_features(window: np.ndarray, gsr_sr: float = 4.0) -> np.ndarray:
    """Extract 9 GSR features from one window."""
    t = _gsr_time_axis(window, gsr_sr)
    slope, _, _, _, _ = linregress(t, window) if len(window) >= 2 else (0.0, 0, 0, 0, 0)
    peaks, props = scipy_signal.find_peaks(
        window, prominence=0.05, distance=int(gsr_sr * 1.0)
    )
    prom = props.get("prominences", np.array([]))
    peak_amp = float(np.mean(prom)) if len(prom) else 0.0
    diff = np.diff(window)
    return np.array(
        [
            float(np.mean(window)),
            float(np.std(window)),
            float(np.max(window)),
            float(np.min(window)),
            float(slope),
            float(len(peaks)),
            peak_amp,
            float(np.mean(diff)) if len(diff) else 0.0,
            float(np.std(diff)) if len(diff) else 0.0,
        ],
        dtype=np.float64,
    )


def _detect_rr_intervals(ecg_window: np.ndarray, ecg_sr: float = 64.0) -> np.ndarray:
    """Return RR intervals in seconds from an ECG window."""
    nyq = ecg_sr / 2.0
    low = 0.5 / nyq
    high = min(40.0 / nyq, 0.99)
    try:
        b, a = scipy_signal.butter(4, [low, high], btype="band")
        filtered = scipy_signal.filtfilt(b, a, ecg_window)
    except Exception:
        filtered = ecg_window
    min_distance = int(ecg_sr * 0.3)
    peaks, _ = scipy_signal.find_peaks(filtered, distance=min_distance)
    if len(peaks) < 3:
        return np.array([])
    rr_samples = np.diff(peaks)
    rr_seconds = rr_samples / ecg_sr
    return rr_seconds[(rr_seconds >= 0.3) & (rr_seconds <= 2.0)]


def extract_hrv_features(ecg_window: np.ndarray, ecg_sr: float = 64.0) -> np.ndarray:
    """Extract 6 HRV features from one ECG window."""
    rr = _detect_rr_intervals(ecg_window, ecg_sr)
    if len(rr) == 0:
        return np.zeros(6, dtype=np.float64)
    mean_hr = 60.0 / np.mean(rr)
    instant_hr = 60.0 / rr
    std_hr = float(np.std(instant_hr))
    if len(rr) >= 2:
        diff_rr = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff_rr**2)))
        sdnn = float(np.std(rr))
        pnn50 = float(np.sum(np.abs(diff_rr) > 0.05) / max(len(diff_rr), 1))
    else:
        rmssd = sdnn = pnn50 = 0.0
    lf_hf = hrv_lf_hf_ratio(rr, ecg_sr)
    return np.array([mean_hr, std_hr, rmssd, sdnn, pnn50, lf_hf], dtype=np.float64)


def hrv_lf_hf_ratio(rr: np.ndarray, ecg_sr: float = 64.0) -> float:
    """LF/HF ratio; returns -1 if unreliable."""
    if len(rr) < 10:
        return -1.0
    rr_times = np.cumsum(rr)
    total_time = rr_times[-1]
    uniform_t = np.arange(0, total_time, 1.0 / 4.0)
    uniform_rr = np.interp(uniform_t, rr_times, rr)
    n = len(uniform_rr)
    freqs = np.fft.rfftfreq(n, d=1.0 / 4.0)
    psd = np.abs(np.fft.rfft(uniform_rr - np.mean(uniform_rr))) ** 2
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    lf_power = _trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 1e-10
    hf_power = _trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 1e-10
    if hf_power < 1e-10:
        return -1.0
    return float(lf_power / hf_power)


def _hrv_scalar_stream_rmssd(hrv_win: np.ndarray) -> float:
    """
    RMS of successive diffs of scalar HRV samples — same construction as
    ``scentsation_hub.compute_features`` (not ECG RR-interval RMSSD).
    """
    v = hrv_win[hrv_win >= 0]
    if len(v) >= 2:
        d = np.diff(v.astype(np.float64))
        return float(np.sqrt(np.mean(d**2)))
    if len(v) == 1:
        return float(abs(v[0]))
    return 0.0


def extract_features(
    gsr_window: np.ndarray,
    ecg_window: np.ndarray,
    gsr_sr: float = 4.0,
    ecg_sr: float = 64.0,
) -> np.ndarray:
    """Full 15-D feature vector for one paired window."""
    g = extract_gsr_features(gsr_window, gsr_sr)
    h = extract_hrv_features(ecg_window, ecg_sr)
    out = np.concatenate([g, h])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features_from_raw_csv(df, gsr_col: str = "gsr_raw", hr_col: str = "hr_raw", hrv_col: str = "hrv_raw"):
    """Rolling-window features from tabular HR/HRV/GSR columns (custom CSV)."""
    gsr_vals = df[gsr_col].values.astype(np.float64)
    hr_vals = df[hr_col].values.astype(np.float64)
    hrv_vals = df[hrv_col].values.astype(np.float64)
    n = len(gsr_vals)
    window_size = min(40, max(8, n))
    if n < window_size:
        pad = window_size - n
        gsr_vals = np.pad(gsr_vals, (0, pad), mode="edge")
        hr_vals = np.pad(hr_vals, (0, pad), mode="edge")
        hrv_vals = np.pad(hrv_vals, (0, pad), mode="edge")
    step = max(1, window_size // 2)
    all_features = []
    for start in range(0, len(gsr_vals) - window_size + 1, step):
        gsr_win = gsr_vals[start : start + window_size]
        hr_win = hr_vals[start : start + window_size]
        hrv_win = hrv_vals[start : start + window_size]
        gsr_feats = extract_gsr_features(gsr_win)
        valid_hr = hr_win[hr_win > 0]
        mean_hr = float(np.mean(valid_hr)) if len(valid_hr) else 0.0
        std_hr = float(np.std(valid_hr)) if len(valid_hr) else 0.0
        hrv_rmssd_stream = _hrv_scalar_stream_rmssd(hrv_win)
        if mean_hr > 0:
            mean_rr = 60.0 / mean_hr
            rr_sim = np.full(len(valid_hr), mean_rr) + np.random.normal(0, 0.02, len(valid_hr))
            sdnn = float(np.std(rr_sim))
            diff_rr = np.abs(np.diff(rr_sim))
            pnn50 = float(np.sum(diff_rr > 0.05) / max(len(diff_rr), 1))
            lf_hf = -1.0
        else:
            sdnn = pnn50 = 0.0
            lf_hf = -1.0
        hrv_feats = np.array([mean_hr, std_hr, hrv_rmssd_stream, sdnn, pnn50, lf_hf])
        all_features.append(np.concatenate([gsr_feats, hrv_feats]))
    return np.array(all_features)


def extract_features_and_labels_from_raw_csv(
    df,
    label_col: str = "label_int",
    gsr_col: str = "gsr_raw",
    hr_col: str = "hr_raw",
    hrv_col: str = "hrv_raw",
):
    """Same windowing as ``extract_features_from_raw_csv`` plus majority label per window."""
    if label_col not in df.columns:
        raise ValueError(f"Missing {label_col}")
    lab = df[label_col].values.astype(int)
    gsr_vals = df[gsr_col].values.astype(np.float64)
    hr_vals = df[hr_col].values.astype(np.float64)
    hrv_vals = df[hrv_col].values.astype(np.float64)
    n = len(gsr_vals)
    window_size = min(40, max(8, n))
    if n < window_size:
        pad = window_size - n
        gsr_vals = np.pad(gsr_vals, (0, pad), mode="edge")
        hr_vals = np.pad(hr_vals, (0, pad), mode="edge")
        hrv_vals = np.pad(hrv_vals, (0, pad), mode="edge")
        lab = np.pad(lab, (0, pad), mode="edge")
    step = max(1, window_size // 2)
    feats = []
    y = []
    for start in range(0, len(gsr_vals) - window_size + 1, step):
        gsr_win = gsr_vals[start : start + window_size]
        hr_win = hr_vals[start : start + window_size]
        hrv_win = hrv_vals[start : start + window_size]
        lw = lab[start : start + window_size]
        gsr_feats = extract_gsr_features(gsr_win)
        valid_hr = hr_win[hr_win > 0]
        mean_hr = float(np.mean(valid_hr)) if len(valid_hr) else 0.0
        std_hr = float(np.std(valid_hr)) if len(valid_hr) else 0.0
        hrv_rmssd_stream = _hrv_scalar_stream_rmssd(hrv_win)
        if mean_hr > 0:
            mean_rr = 60.0 / mean_hr
            rr_sim = np.full(len(valid_hr), mean_rr) + np.random.normal(0, 0.02, len(valid_hr))
            sdnn = float(np.std(rr_sim))
            diff_rr = np.abs(np.diff(rr_sim))
            pnn50 = float(np.sum(diff_rr > 0.05) / max(len(diff_rr), 1))
            lf_hf = -1.0
        else:
            sdnn = pnn50 = 0.0
            lf_hf = -1.0
        hrv_feats = np.array([mean_hr, std_hr, hrv_rmssd_stream, sdnn, pnn50, lf_hf])
        feats.append(np.concatenate([gsr_feats, hrv_feats]))
        y.append(int(np.bincount(lw).argmax()))
    return np.array(feats), np.array(y, dtype=int)
