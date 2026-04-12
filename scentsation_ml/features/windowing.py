"""
Sliding window segmentation for physiological signals (GSR + ECG).
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def sliding_window(
    signal: np.ndarray,
    sample_rate: float,
    window_size_sec: float = 10.0,
    overlap_ratio: float = 0.5,
) -> List[np.ndarray]:
    """Split a 1-D signal into overlapping windows."""
    if window_size_sec <= 0:
        raise ValueError("window_size_sec must be positive")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")

    window_samples = int(window_size_sec * sample_rate)
    step_samples = max(1, int(window_samples * (1.0 - overlap_ratio)))
    total_samples = len(signal)
    windows: List[np.ndarray] = []

    for start in range(0, total_samples - window_samples + 1, step_samples):
        windows.append(signal[start : start + window_samples])

    remainder_start = len(windows) * step_samples
    if remainder_start < total_samples:
        tail = signal[remainder_start:]
        if len(tail) > window_samples * 0.5:
            padded = np.pad(tail, (0, window_samples - len(tail)), mode="edge")
            windows.append(padded)

    logger.debug(
        "Windowing: %d samples @ %.1f Hz → %d windows",
        total_samples,
        sample_rate,
        len(windows),
    )
    return windows


def assign_window_labels(
    sample_labels: np.ndarray,
    sample_rate: float,
    window_size_sec: float = 10.0,
    overlap_ratio: float = 0.5,
) -> np.ndarray:
    """Assign one label per window via majority vote of sample labels."""
    window_samples = int(window_size_sec * sample_rate)
    step_samples = max(1, int(window_samples * (1.0 - overlap_ratio)))
    total = len(sample_labels)
    labels: List[int] = []

    for start in range(0, total - window_samples + 1, step_samples):
        window_labels = sample_labels[start : start + window_samples]
        valid = window_labels[window_labels >= 0]
        if len(valid) == 0:
            labels.append(-1)
        else:
            unique, counts = np.unique(valid, return_counts=True)
            labels.append(int(unique[np.argmax(counts)]))

    remainder_start = len(labels) * step_samples
    if remainder_start < total:
        tail_labels = sample_labels[remainder_start:]
        if len(tail_labels) > window_samples * 0.5:
            valid = tail_labels[tail_labels >= 0]
            if len(valid) == 0:
                labels.append(-1)
            else:
                unique, counts = np.unique(valid, return_counts=True)
                labels.append(int(unique[np.argmax(counts)]))

    return np.array(labels, dtype=int)


def segment_gsr_ecg_pair(
    gsr_signal: np.ndarray,
    ecg_signal: np.ndarray,
    gsr_sr: float,
    ecg_sr: float,
    sample_labels: np.ndarray,
    window_size_sec: float = 10.0,
    overlap_ratio: float = 0.5,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Segment aligned GSR/ECG windows and window labels (ECG label resolution)."""
    gsr_windows = sliding_window(gsr_signal, gsr_sr, window_size_sec, overlap_ratio)
    ecg_windows = sliding_window(ecg_signal, ecg_sr, window_size_sec, overlap_ratio)
    window_labels = assign_window_labels(
        sample_labels, ecg_sr, window_size_sec, overlap_ratio
    )
    min_len = min(len(gsr_windows), len(ecg_windows), len(window_labels))
    return (
        gsr_windows[:min_len],
        ecg_windows[:min_len],
        window_labels[:min_len],
    )
