#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Scentsation Hub — Central Orchestration Layer
=============================================
A machine that reads GSR + PPG biosignals, classifies physiological state
via a lightweight sklearn classifier, converses with the user via ZhipuAI GLM-4,
and triggers scent pumps to dispense a personalized fragrance blend.

Requirements (pip install -r requirements.txt):
    pyserial
    numpy
    scikit-learn
    joblib
    pyyaml
    openai
    python-dotenv
    rich
    speech_recognition  # optional, for microphone input

Example .env (do not commit secrets):
    ZHIPU_API_KEY=your_zhipu_api_key_here

    Classifier note:
    The hub expects a joblib dict ``{"model", "scaler", "feature_names"}`` (see
    ``config/hub_feature_names.json``) unless ``--allow-bare-model`` or
    ``SCENTSATION_ALLOW_BARE_MODEL=1``. Dict payloads **must** include ``feature_names``
    matching the canonical list.
    The model must accept **6** features per sample:
    ``[gsr_mean, gsr_max, gsr_std, gsr_slope, hr_mean, hrv_rmssd]`` (see
    ``compute_features()``). This matches the MVP hub specification; the
    separate ``scentsation_ml`` trainer exports 15-D features — retrain or
    export a 6-D model for on-device inference if you need parity.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import queue
import random
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from joblib import load as joblib_load
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    import serial
except ImportError:
    serial = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    import speech_recognition as sr  # type: ignore
except ImportError:
    sr = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Scentsation")

console = Console()

LINE_RE = re.compile(
    r"^\s*GSR:\s*([+-]?\d*\.?\d+)\s*,\s*HR:\s*([+-]?\d*\.?\d+)\s*,\s*HRV:\s*([+-]?\d*\.?\d+)\s*$",
    re.IGNORECASE,
)

POSITIVE_FOCUS_KEYWORDS = [
    "sharp", "alert", "focused", "awake", "energized", "clear", "fresh",
    "invigorated", "motivated", "bright", "driven", "alive", "stimulated",
    "attentive", "engaged", "productive", "zesty", "cool", "uplifted",
    "confident", "ready", "strong", "dynamic", "crisp", "good", "great",
    "nice", "love", "like", "amazing", "wonderful", "excellent", "fantastic",
    "pleasant", "refreshing", "soothing", "centered", "balanced",
]

NEGATIVE_KEYWORDS = [
    "distracted", "headache", "nausea", "dizzy", "annoyed", "bored",
    "sleepy", "tired", "drowsy", "irritated", "uncomfortable", "bad",
    "terrible", "horrible", "hate", "dislike", "awful", "stuffy",
    "overwhelming", "too strong", "suffocating", "choking", "anxious",
    "nervous", "uneasy", "restless", "confused", "foggy", "heavy",
    "dull", "flat", "nothing", "no effect", "bland",
]

CLASS_ORDER = ["NEUTRAL", "RELAXED", "STRESSED", "FOCUSED"]

# Must match ``compute_features()`` output length (streaming hub — not ``scentsation_ml`` 15-D windows).
HUB_FEATURE_DIM = 6

_FALLBACK_HUB_FEATURE_NAMES: Tuple[str, ...] = (
    "gsr_mean",
    "gsr_max",
    "gsr_std",
    "gsr_slope",
    "hr_mean",
    "hrv_rmssd",
)


def _hub_feature_names_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "hub_feature_names.json")


def load_hub_feature_names_tuple() -> Tuple[str, ...]:
    """
    Load ordered feature names from ``config/hub_feature_names.json`` (single source of truth).

    Falls back to :data:`_FALLBACK_HUB_FEATURE_NAMES` if the file is missing, unless
    ``SCENTSATION_REQUIRE_HUB_FEATURE_JSON=1`` (strict mode for demos/CI) — then raises.
    """
    path = _hub_feature_names_config_path()
    strict = os.getenv("SCENTSATION_REQUIRE_HUB_FEATURE_JSON", "").strip() == "1"
    if not os.path.isfile(path):
        if strict:
            raise FileNotFoundError(
                f"Strict mode (SCENTSATION_REQUIRE_HUB_FEATURE_JSON=1): missing {path}"
            )
        logger.warning(
            "config/hub_feature_names.json not found at %s — using built-in fallback names",
            path,
        )
        return _FALLBACK_HUB_FEATURE_NAMES
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != HUB_FEATURE_DIM:
        raise ValueError(
            f"{path} must be a JSON array of length {HUB_FEATURE_DIM}, got {data!r}"
        )
    return tuple(str(x) for x in data)


# Canonical 6-D order (from JSON when present); used by ``compute_features`` / validation.
HUB_FEATURE_NAMES: Tuple[str, ...] = load_hub_feature_names_tuple()

# Warn once if ESP32 input thread hits this many consecutive read errors (USB/cable stress).
ESP_SERIAL_ERROR_WARN_THRESHOLD = 80
# After this many consecutive read/write failures, try to close/reopen the port (USB blip recovery).
SERIAL_IO_RECONNECT_THRESHOLD = 25


class SessionState(Enum):
    """States for the scent-test session state machine."""

    IDLE = auto()
    CALIBRATION = auto()
    SPRAYING = auto()
    OBSERVING = auto()
    ASKING_USER = auto()
    BLEND_SELECT = auto()
    COMPLETE = auto()


@dataclass
class Config:
    """Central configuration for the Scentsation hub."""

    esp32_port: str
    arduino_port: str
    baud_rate: int = 115200
    serial_timeout: float = 1.0
    calibration_duration: int = 30
    spray_duration: int = 5
    observe_duration: int = 10
    scent_notes: List[str] = field(default_factory=list)
    model_path: str = "models/classifier.joblib"
    buffer_window_sec: float = 10.0
    use_llm: bool = True
    mock_mode: bool = False
    no_dashboard: bool = False
    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    zhipu_model: str = "glm-4"
    weight_ml: float = 0.6
    weight_sentiment: float = 0.4
    log_dir: str = "logs"

    def __post_init__(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_log_path = os.path.join(self.log_dir, f"session_{ts}.csv")


@dataclass
class SensorReading:
    """A single parsed sensor reading from the ESP32."""

    gsr: float
    hr: float
    hrv: float
    timestamp: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        """Return True if GSR and HR are usable (non-negative)."""
        return self.gsr >= 0 and self.hr >= 0


def parse_sensor_line(line: str) -> Optional[SensorReading]:
    """Parse one ``GSR:x,HR:y,HRV:z`` line into a :class:`SensorReading`."""
    m = LINE_RE.match(line.strip())
    if not m:
        return None
    gsr, hr, hrv = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return SensorReading(gsr=gsr, hr=hr, hrv=hrv)


def expected_n_features_in(model: Any) -> Optional[int]:
    """
    Return sklearn ``n_features_in_`` when set, else ``None`` (skip strict checks).
    """
    return getattr(model, "n_features_in_", None)


def _transform_x_for_predict(model: Any, scaler: Any, x: np.ndarray) -> np.ndarray:
    """Apply the same scaling path as :func:`apply_model` (``x`` shape ``(1, HUB_FEATURE_DIM)``)."""
    from sklearn.pipeline import Pipeline

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is not None and not isinstance(model, Pipeline):
        return scaler.transform(x)
    return x


def probe_predict_hub(model: Any, scaler: Any) -> None:
    """
    Run ``predict`` on a zero probe vector using the hub transform path.

    Catches checkpoints with missing ``n_features_in_`` or wrong input shape.
    """
    x = np.zeros((1, HUB_FEATURE_DIM), dtype=np.float64)
    try:
        xt = _transform_x_for_predict(model, scaler, x)
        model.predict(xt)
    except Exception as e:
        raise ValueError(
            "Model is not compatible with the hub (6-D streaming features). "
            "Train a 6-D classifier or use a scentsation_ml adapter — "
            f"probe predict failed: {e!r}"
        ) from e


def preflight_serial_ports(cfg: Config) -> None:
    """
    Open each USB serial path briefly to fail fast if ports are wrong or missing.

    Does not start :class:`SerialManager` threads; real session opens ports again.
    """
    if serial is None:
        console.print(
            Panel(
                "pyserial is not installed — cannot use USB mode. "
                "Install with `pip install pyserial` or use --mock-mode.",
                title="Serial preflight failed",
                border_style="red",
            )
        )
        sys.exit(1)
    pairs: Tuple[Tuple[str, str], ...] = (
        (cfg.esp32_port, "ESP32 (sensors)"),
        (cfg.arduino_port, "Arduino (pumps)"),
    )
    for path, label in pairs:
        try:
            ser = serial.Serial(path, cfg.baud_rate, timeout=0.2)
            try:
                ser.close()
            except Exception:
                pass
        except Exception as e:
            logger.error("Serial preflight failed for %s (%s): %s", label, path, e)
            console.print(
                Panel(
                    f"Could not open [bold]{path}[/bold] ({label}).\n\n"
                    "Check USB cables, permissions, and that the device is enumerated. "
                    "Confirm [bold]ESP32_PORT[/bold] and [bold]ARDUINO_PORT[/bold] "
                    "match your system (e.g. `ls /dev/tty.*` on macOS/Linux).\n\n"
                    f"Error: {e!r}",
                    title="Serial preflight failed",
                    border_style="red",
                )
            )
            sys.exit(1)


def validate_model_for_hub(
    model: Any,
    scaler: Any,
    n_expected: int,
    payload: Optional[Dict[str, Any]] = None,
    *,
    allow_bare_model: bool = False,
) -> None:
    """
    Ensure the loaded estimator expects the same input dimension as ``compute_features``.

    ``scentsation_ml`` trains 15-D window models; the hub uses 6-D streaming features.
    Loading a 15-D checkpoint without an adapter must fail with a clear error.

    Dict joblibs must include ``feature_names`` matching :data:`HUB_FEATURE_NAMES`.
    Bare estimators are rejected unless ``allow_bare_model`` or ``SCENTSATION_ALLOW_BARE_MODEL=1``.
    """
    env_bare = os.getenv("SCENTSATION_ALLOW_BARE_MODEL", "").strip() == "1"
    if payload is None:
        if not (allow_bare_model or env_bare):
            raise ValueError(
                "Loaded joblib is a bare estimator with no metadata. "
                "Save a dict payload with keys model, scaler (optional), and "
                "feature_names matching config/hub_feature_names.json — see "
                "scentsation_ml.export.dump_hub_joblib. "
                "For legacy files only, pass --allow-bare-model or set "
                "SCENTSATION_ALLOW_BARE_MODEL=1."
            )
        logger.error(
            "Loading bare sklearn estimator without feature_names metadata (legacy mode).",
        )
        console.print(
            Panel(
                "Bare joblib: no dict / no feature_names. Do not use for production or "
                "competition demos — train/export with scentsation_ml.export.dump_hub_joblib.",
                title="WARNING",
                border_style="red",
            )
        )
    else:
        if "feature_names" not in payload or payload["feature_names"] is None:
            raise ValueError(
                "Joblib dict must include a non-null 'feature_names' list matching "
                "config/hub_feature_names.json. Use scentsation_ml.export.build_hub_joblib_payload."
            )
        got = [str(x) for x in payload["feature_names"]]
        exp = list(HUB_FEATURE_NAMES)
        if got != exp:
            raise ValueError(
                f"Joblib feature_names do not match the hub streaming order. "
                f"Expected {exp!r}, got {got!r}. "
                "Train/export with the same 6-D definitions as compute_features()."
            )
    n = expected_n_features_in(model)
    if n is not None and n != n_expected:
        raise ValueError(
            f"Model expects {n} features but the hub produces {n_expected} "
            f"(see compute_features / HUB_FEATURE_DIM). "
            "scentsation_ml exports 15-D window models — train or export a 6-D "
            "classifier for this script, or add an adapter."
        )
    probe_predict_hub(model, scaler)


def load_model_payload(path: str) -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
    """
    Load ``(model, scaler, payload_dict)`` from a ``.joblib`` file.

    Supports dict payloads ``{"model", "scaler", ...}`` or a bare estimator.
    ``payload_dict`` is the full dict when present, else ``None``.
    """
    raw = joblib_load(path)
    if isinstance(raw, dict) and "model" in raw:
        return raw["model"], raw.get("scaler"), raw
    return raw, None, None


def ensure_stub_model(path: str) -> None:
    """
    If ``path`` is missing, write a tiny 6-feature sklearn model for dry-runs.

    This keeps ``--mock-mode`` runnable before you train a real classifier.
    """
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    rng = np.random.RandomState(42)
    X = rng.randn(80, 6)
    y = rng.randint(0, 4, size=80)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
    )
    pipe.fit(X, y)
    joblib.dump(
        {
            "model": pipe,
            "scaler": None,
            "config": {"classes": CLASS_ORDER},
            "feature_names": list(HUB_FEATURE_NAMES),
        },
        path,
    )
    logger.warning("Wrote stub classifier to %s (replace with a real model).", path)


def apply_model(model: Any, scaler: Any, features: np.ndarray) -> str:
    """
    Run inference and return a string label among :data:`CLASS_ORDER`.

    Supports integer labels ``0..3`` (mapped via :data:`CLASS_ORDER`) or string
    labels that already match the class names.
    """
    if features.size != HUB_FEATURE_DIM:
        raise ValueError(
            f"Expected {HUB_FEATURE_DIM} features, got {features.size} — check compute_features."
        )
    features = np.nan_to_num(
        np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    x = features.reshape(1, -1)
    n_in = expected_n_features_in(model)
    if n_in is not None and n_in != HUB_FEATURE_DIM:
        raise ValueError(
            f"Model n_features_in_={n_in} does not match hub HUB_FEATURE_DIM={HUB_FEATURE_DIM}."
        )
    x = _transform_x_for_predict(model, scaler, x)
    pred = model.predict(x)[0]
    if isinstance(pred, str):
        return pred if pred in CLASS_ORDER else "NEUTRAL"
    try:
        pv = float(np.asarray(pred, dtype=np.float64).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return "NEUTRAL"
    if not np.isfinite(pv):
        return "NEUTRAL"
    try:
        idx = int(pv)
    except (TypeError, ValueError, OverflowError):
        return "NEUTRAL"
    if 0 <= idx < len(CLASS_ORDER):
        return CLASS_ORDER[idx]
    return "NEUTRAL"


def compute_features(readings: Deque[SensorReading]) -> np.ndarray:
    """
    Compute the 6-D feature vector from the rolling buffer.

    Returns:
        ``numpy.ndarray`` of shape ``(6,)``:
        ``[gsr_mean, gsr_max, gsr_std, gsr_slope, hr_mean, hrv_rmssd]``.

    The last column is **not** ECG RR-based RMSSD: it is the RMS of successive
    differences of the hub's scalar HRV stream (same 4 Hz samples as ``esp32_sensors``
    ``currentHRV``), aligned with training exports that use the same definition.
    """
    if len(readings) < 2:
        return np.zeros(6, dtype=np.float64)

    gsr = np.array([r.gsr for r in readings], dtype=np.float64)
    hr = np.array([r.hr for r in readings], dtype=np.float64)
    hrv = np.array([r.hrv for r in readings], dtype=np.float64)

    gsr_mean = float(np.mean(gsr))
    gsr_max = float(np.max(gsr))
    gsr_std = float(np.std(gsr))
    t = np.arange(len(gsr), dtype=np.float64)
    if len(gsr) >= 2 and np.std(t) > 1e-9:
        gsr_slope = float(np.polyfit(t, gsr, 1)[0])
    else:
        gsr_slope = 0.0

    valid_hr = hr[hr >= 0]
    hr_mean = float(np.mean(valid_hr)) if len(valid_hr) else 0.0

    valid_hrv = hrv[hrv >= 0]
    if len(valid_hrv) >= 2:
        # RMS of successive diffs of the scalar HRV stream (hub-aligned, not lab RR).
        diffs = np.diff(valid_hrv.astype(np.float64))
        hrv_rmssd = float(np.sqrt(np.mean(diffs**2)))
    elif len(valid_hrv) == 1:
        hrv_rmssd = float(abs(valid_hrv[0]))
    else:
        # PPG offline: ESP sends HR/HRV = -1 — do not mix sentinels into spread stats.
        hrv_rmssd = 0.0

    return np.array(
        [gsr_mean, gsr_max, gsr_std, gsr_slope, hr_mean, hrv_rmssd],
        dtype=np.float64,
    )


def sentiment_score(text: str) -> float:
    """Return a score in ``[0, 1]`` from keyword heuristics."""
    low = text.lower()
    pos = sum(1 for w in POSITIVE_FOCUS_KEYWORDS if w in low)
    neg = sum(1 for w in NEGATIVE_KEYWORDS if w in low)
    if pos + neg == 0:
        return 0.5
    return pos / (pos + neg)


class SerialManager:
    """
    Manage ESP32 (sensor) and Arduino (pump) serial ports in daemon threads.

    * **input** mode: read lines and push :class:`SensorReading` into a queue.
    * **output** mode: send ``PUMP:...`` commands and read ``ACK`` lines back.
    """

    def __init__(self, config: Config, mode: str):
        """Initialise the serial manager."""
        self.config = config
        self.mode = mode
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._latest_lock = threading.Lock()
        self._rx_lines: Optional[queue.Queue[str]] = None

        if mode == "input":
            self.port = config.esp32_port
        elif mode == "output":
            self.port = config.arduino_port
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.queue: "queue.Queue[SensorReading]" = queue.Queue(maxsize=5000)
        self.latest: Optional[SensorReading] = None
        self.last_ack: str = ""
        self.connected = False
        self._ser: Any = None
        self._join_timeout_sec = 3.0
        self._consec_serial_errors = 0
        self._queue_drop_logged = False
        if mode == "output":
            self._rx_lines = queue.Queue(maxsize=256)

    def _open_serial(self) -> Any:
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        return serial.Serial(
            self.port,
            self.config.baud_rate,
            timeout=self.config.serial_timeout,
        )

    def _ensure_serial_open(self) -> bool:
        """Open the port if missing or closed (used by reader loops and ``start``)."""
        if self._stop.is_set():
            return False
        if self._ser is not None and getattr(self._ser, "is_open", False):
            return True
        try:
            self._ser = self._open_serial()
            return True
        except Exception as e:
            logger.warning("Serial open failed for %s: %s", self.port, e)
            return False

    def _reopen_serial_after_io_error(self) -> bool:
        """Best-effort ``close`` + ``open`` after USB disconnect or driver errors."""
        if serial is None or self._stop.is_set():
            return False
        logger.warning("Reconnecting serial port %s after I/O errors", self.port)
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        time.sleep(0.2)
        ok = self._ensure_serial_open()
        if ok:
            logger.info("Serial port active again: %s", self.port)
        return ok

    def get_latest(self) -> Optional[SensorReading]:
        """Thread-safe snapshot of the most recent sensor line (input mode only)."""
        if self.mode != "input":
            return None
        with self._latest_lock:
            return self.latest

    def serial_error_streak(self) -> int:
        """Consecutive serial read errors on the background thread (diagnostics)."""
        return self._consec_serial_errors

    def start(self) -> None:
        """Open the port and start the background thread."""
        if not self._ensure_serial_open():
            raise RuntimeError(f"Could not open serial port {self.port!r}")
        self.connected = True
        if self.mode == "input":
            self._thread = threading.Thread(target=self._input_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._output_idle_loop, daemon=True)
        self._thread.start()

    def _output_idle_loop(self) -> None:
        """Arduino thread: block on reads so late ACKs do not get lost."""
        logger.info("[Arduino] Listening on %s", self.port)
        while not self._stop.is_set():
            if not self._ensure_serial_open():
                time.sleep(0.5)
                continue
            try:
                line = self._ser.readline()
                if line:
                    line_str = line.decode("utf-8", errors="replace").strip()
                    self.last_ack = line_str
                    if self._rx_lines is not None:
                        try:
                            self._rx_lines.put_nowait(line_str)
                        except queue.Full:
                            _ = self._rx_lines.get_nowait()
                            self._rx_lines.put_nowait(line_str)
                    self._consec_serial_errors = 0
            except Exception as e:
                self._consec_serial_errors += 1
                recoverable = isinstance(e, OSError) or (
                    serial is not None and isinstance(e, serial.SerialException)
                )
                if recoverable and self._consec_serial_errors >= SERIAL_IO_RECONNECT_THRESHOLD:
                    if self._reopen_serial_after_io_error():
                        self._consec_serial_errors = 0
                        continue
                if self._consec_serial_errors == 1 or self._consec_serial_errors % 40 == 0:
                    logger.warning(
                        "[Arduino] read error (%s consecutive): %s",
                        self._consec_serial_errors,
                        e,
                    )
                time.sleep(0.05)

    def _input_loop(self) -> None:
        """ESP32 thread: parse ``GSR:,HR:,HRV:`` lines."""
        logger.info("[ESP32] Listening on %s", self.port)
        while not self._stop.is_set():
            if not self._ensure_serial_open():
                time.sleep(0.5)
                continue
            try:
                line = self._ser.readline()
                if not line:
                    continue
                text = line.decode("utf-8", errors="replace").strip()
                reading = parse_sensor_line(text)
                if reading is None:
                    logger.debug("Malformed line: %r", text[:120])
                    continue
                with self._latest_lock:
                    self.latest = reading
                try:
                    self.queue.put_nowait(reading)
                except queue.Full:
                    if not self._queue_drop_logged:
                        logger.warning(
                            "Sensor queue full on %s; dropping oldest samples "
                            "(main thread may be blocked — check LLM or UI stalls)",
                            self.port,
                        )
                        self._queue_drop_logged = True
                    _ = self.queue.get_nowait()
                    self.queue.put_nowait(reading)
                self._consec_serial_errors = 0
            except Exception as e:
                self._consec_serial_errors += 1
                recoverable = isinstance(e, OSError) or (
                    serial is not None and isinstance(e, serial.SerialException)
                )
                if recoverable and self._consec_serial_errors >= SERIAL_IO_RECONNECT_THRESHOLD:
                    if self._reopen_serial_after_io_error():
                        self._consec_serial_errors = 0
                        continue
                if self._consec_serial_errors == 1 or self._consec_serial_errors % 40 == 0:
                    logger.warning(
                        "[ESP32] read error (%s consecutive): %s",
                        self._consec_serial_errors,
                        e,
                    )
                time.sleep(0.02)

    def send_command(self, cmd: str, expect_ack: bool = True, ack_timeout: float = 0.5) -> str:
        """Send a line to the Arduino and optionally wait for an ``ACK`` line."""
        if self.mode != "output" or not self._ser or not self._ser.is_open:
            return ""
        line = cmd if cmd.endswith("\n") else cmd + "\n"
        with self._write_lock:
            if self._rx_lines is not None:
                while True:
                    try:
                        self._rx_lines.get_nowait()
                    except queue.Empty:
                        break
            self._ser.write(line.encode("utf-8"))
            self._ser.flush()
        if not expect_ack:
            return "SKIP_ACK"
        if self._rx_lines is None:
            return "TIMEOUT"
        deadline = time.time() + ack_timeout
        max_lines = 256
        n_seen = 0
        while time.time() < deadline and n_seen < max_lines:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                line_str = self._rx_lines.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            n_seen += 1
            if line_str.startswith("ACK:") or line_str.startswith("ERROR:"):
                return line_str
        logger.warning("No ACK for cmd %r (last=%r)", cmd.strip(), self.last_ack)
        return "TIMEOUT"

    def close(self) -> None:
        """Stop threads and close the serial port."""
        self._stop.set()
        if self._ser and self._ser.is_open:
            self._ser.close()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout_sec)
            if self._thread.is_alive():
                logger.warning(
                    "Serial reader thread for %s did not exit within %ss",
                    self.port,
                    self._join_timeout_sec,
                )
        self.connected = False


class RollingBuffer:
    """Time-based rolling buffer of :class:`SensorReading` instances."""

    def __init__(self, window_sec: float) -> None:
        """Initialise with window length in seconds."""
        self.window_sec = window_sec
        self.items: Deque[SensorReading] = deque()

    def push(self, r: SensorReading) -> None:
        """Append a reading and drop samples older than ``window_sec``."""
        self.items.append(r)
        cutoff = time.time() - self.window_sec
        while self.items and self.items[0].timestamp < cutoff:
            self.items.popleft()

    def clear(self) -> None:
        """Remove all buffered readings."""
        self.items.clear()


def build_dashboard(
    state: SessionState,
    scent_name: str,
    latest: Optional[SensorReading],
    prediction: str,
    buf_len: int,
    buf_cap: int,
) -> Table:
    """Build a small Rich table for the live terminal dashboard."""
    t = Table(title="Scentsation Hub", show_header=True, header_style="bold cyan")
    t.add_column("Field", style="dim")
    t.add_column("Value")
    t.add_row("State", state.name)
    t.add_row("Active scent", scent_name or "—")
    if latest:
        t.add_row("GSR (µS)", f"{latest.gsr:.2f}")
        t.add_row("HR (bpm)", f"{latest.hr:.1f}")
        t.add_row("HRV", f"{latest.hrv:.1f}")
    else:
        t.add_row("Sensors", "—")
    t.add_row("ML prediction", prediction)
    t.add_row("Buffer", f"{buf_len} / ~{buf_cap} samples")
    return t


def ask_user(question: str, use_speech: bool = False) -> str:
    """
    Ask a question on the terminal and return the user's reply.

    If ``use_speech`` is True and ``speech_recognition`` is installed, try the
    microphone first, then fall back to keyboard input.
    """
    console.print(f"[bold green]?[/bold green] {question}")
    if use_speech and sr is not None:
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                console.print("[dim]Listening…[/dim]")
                audio = r.listen(source, timeout=6, phrase_time_limit=12)
            return r.recognize_google(audio)
        except Exception as e:
            console.print(f"[yellow]Speech fallback ({e}). Type your answer:[/yellow]")
    return input("> ").strip()


def make_llm_client(api_key: str, base_url: str) -> Any:
    """Construct an OpenAI-compatible client for Zhipu."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    # Bounded wait so a hung API does not stall the whole session thread.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=45.0)


def llm_reply(
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
) -> str:
    """Send one chat completion and return assistant text."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.6,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("LLM request failed: %s", e)
        return "[Assistant unavailable — continuing without a reply.]"


SYSTEM_PROMPT = (
    "You are a calm, professional scent consultant for a neuroscience-driven "
    "perfume experience. Ask the user about their focus habits, guide them through "
    "the scent test, and keep responses under 3 sentences."
)


def append_csv(path: str, row: Dict[str, Any]) -> None:
    """Append one row to the session CSV (creates header if needed)."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def warn_if_esp_serial_degraded(
    esp: Optional["SerialManager"],
    already_warned: List[bool],
) -> None:
    """Print a one-time console warning if the ESP32 reader reports sustained errors."""
    if esp is None or already_warned[0]:
        return
    if esp.serial_error_streak() < ESP_SERIAL_ERROR_WARN_THRESHOLD:
        return
    already_warned[0] = True
    console.print(
        "[bold yellow]Warning:[/bold yellow] ESP32 serial read errors are high — "
        "the sensor stream may be interrupted. Check the USB cable and port."
    )
    logger.error(
        "ESP32 serial_error_streak=%s (threshold %s)",
        esp.serial_error_streak(),
        ESP_SERIAL_ERROR_WARN_THRESHOLD,
    )


def run_session(cfg: Config, model: Any, scaler: Any) -> None:
    """Run calibration, per-scent spray/observe/ask loop, and blend selection."""
    buf = RollingBuffer(cfg.buffer_window_sec)
    predictions_log: List[str] = []

    esp: Optional[SerialManager] = None
    ard: Optional[SerialManager] = None
    mock_feed: Optional["MockSensorFeed"] = None

    if cfg.mock_mode:
        mock_feed = MockSensorFeed()
        mock_feed.start()
    else:
        esp = SerialManager(cfg, "input")
        ard = SerialManager(cfg, "output")
        esp.start()
        ard.start()

    state = SessionState.IDLE
    current_scent = ""
    live_prediction = "—"
    notes_results: List[Dict[str, Any]] = []
    last_ingested_sig: Optional[Tuple[float, float, float, float]] = None
    esp_serial_warned: List[bool] = [False]

    client = None
    if cfg.use_llm and cfg.zhipu_api_key:
        client = make_llm_client(cfg.zhipu_api_key, cfg.zhipu_base_url)

    def ingest_reading(r: SensorReading) -> None:
        nonlocal live_prediction, last_ingested_sig
        last_ingested_sig = (r.timestamp, r.gsr, r.hr, r.hrv)
        buf.push(r)
        feats = compute_features(buf.items)
        pred = apply_model(model, scaler, feats)
        live_prediction = pred
        predictions_log.append(pred)
        append_csv(cfg.csv_log_path, {
            "time": datetime.now().isoformat(),
            "phase": state.name,
            "gsr": r.gsr,
            "hr": r.hr,
            "hrv": r.hrv,
            "pred": pred,
            "f0": feats[0], "f1": feats[1], "f2": feats[2],
            "f3": feats[3], "f4": feats[4], "f5": feats[5],
        })

    try:
        # --- Opening ---
        if client:
            opening = llm_reply(
                client, cfg.zhipu_model, SYSTEM_PROMPT,
                "Open the session: ask what kind of work they need focus for today.",
            )
            console.print(Panel(opening, title="Assistant"))
            append_csv(cfg.csv_log_path, {"time": datetime.now().isoformat(), "llm": opening})
        ask_user("What kind of work or study do you need focus for today?")

        # --- Calibration ---
        state = SessionState.CALIBRATION
        console.print("[bold]Calibration:[/bold] rest hands on sensors; breathing normally.")
        if client:
            pre = llm_reply(
                client, cfg.zhipu_model, SYSTEM_PROMPT,
                "Say in one short sentence: ask them to rest hands on sensors for 30s.",
            )
            console.print(Panel(pre, title="Assistant"))
        ask_user(
            "Please rest your hands on the sensors and breathe normally for 30 seconds. "
            "Press Enter when you begin.",
        )
        t0 = time.time()
        while time.time() - t0 < cfg.calibration_duration:
            if cfg.mock_mode and mock_feed:
                ingest_reading(mock_feed.next_reading())
            elif esp:
                try:
                    r = esp.queue.get(timeout=0.2)
                    ingest_reading(r)
                except queue.Empty:
                    lr = esp.get_latest()
                    if lr is not None:
                        sig = (lr.timestamp, lr.gsr, lr.hr, lr.hrv)
                        if last_ingested_sig != sig:
                            ingest_reading(lr)
            warn_if_esp_serial_degraded(esp, esp_serial_warned)
            time.sleep(0.05)

        # --- Testing each scent ---
        for idx, name in enumerate(cfg.scent_notes):
            state = SessionState.SPRAYING
            current_scent = name
            pump_num = idx + 1
            cmd = f"PUMP:{pump_num}"
            ack = ""
            if ard:
                ack = ard.send_command(cmd)
            logger.info("Pump command %s → %s", cmd, ack)
            append_csv(cfg.csv_log_path, {"time": datetime.now().isoformat(), "pump_cmd": cmd, "ack": ack})
            if ard and (not ack or ack == "TIMEOUT"):
                console.print(
                    Panel(
                        "[bold red]Pump command did not acknowledge[/bold red] (no reply or TIMEOUT).\n"
                        "Check Arduino USB, correct serial port, and relay power. "
                        "Sending best-effort PUMP:OFF.",
                        title="Serial / pump",
                        border_style="red",
                    )
                )
                logger.error("Pump ACK missing: cmd=%r ack=%r", cmd, ack)
                append_csv(
                    cfg.csv_log_path,
                    {
                        "time": datetime.now().isoformat(),
                        "pump_error": "no_ack",
                        "pump_cmd": cmd,
                        "ack": ack,
                    },
                )
                ard.send_command("PUMP:OFF", expect_ack=False)

            spray_end = time.time() + cfg.spray_duration
            while time.time() < spray_end:
                if cfg.mock_mode and mock_feed:
                    ingest_reading(mock_feed.next_reading())
                elif esp:
                    try:
                        ingest_reading(esp.queue.get(timeout=0.2))
                    except queue.Empty:
                        lr = esp.get_latest()
                        if lr is not None:
                            sig = (lr.timestamp, lr.gsr, lr.hr, lr.hrv)
                            if last_ingested_sig != sig:
                                ingest_reading(lr)
                warn_if_esp_serial_degraded(esp, esp_serial_warned)
                time.sleep(0.05)

            if ard:
                ard.send_command("PUMP:OFF")

            state = SessionState.OBSERVING
            if not cfg.no_dashboard:
                lr = buf.items[-1] if buf.items else (esp.get_latest() if esp else None)
                console.print(
                    build_dashboard(
                        state, name, lr, live_prediction,
                        len(buf.items), max(1, int(cfg.buffer_window_sec * 4)),
                    )
                )
            obs_end = time.time() + cfg.observe_duration
            note_preds: List[str] = []
            while time.time() < obs_end:
                if cfg.mock_mode and mock_feed:
                    r = mock_feed.next_reading()
                    ingest_reading(r)
                    note_preds.append(live_prediction)
                elif esp:
                    try:
                        r = esp.queue.get(timeout=0.2)
                        ingest_reading(r)
                        note_preds.append(live_prediction)
                    except queue.Empty:
                        lr = esp.get_latest()
                        if lr is not None:
                            sig = (lr.timestamp, lr.gsr, lr.hr, lr.hrv)
                            if last_ingested_sig != sig:
                                ingest_reading(lr)
                                note_preds.append(live_prediction)
                warn_if_esp_serial_degraded(esp, esp_serial_warned)
                time.sleep(0.05)

            ml_match = 1.0 if any(p == "FOCUSED" for p in note_preds[-20:]) else 0.0

            state = SessionState.ASKING_USER
            q = (
                f"You just experienced {name}. How did that make you feel? "
                "One or two words — sharp, calm, distracted, energized, etc."
            )
            if client:
                q_llm = llm_reply(client, cfg.zhipu_model, SYSTEM_PROMPT, q)
                console.print(Panel(q_llm, title="Assistant"))
            reply = ask_user(q)
            sent = sentiment_score(reply)
            final_score = cfg.weight_ml * ml_match + cfg.weight_sentiment * sent
            notes_results.append({
                "scent_id": name,
                "ml_match": ml_match,
                "user_sentiment": sent,
                "final_score": final_score,
                "reply": reply,
                "ml_pred_sample": note_preds[-1] if note_preds else live_prediction,
            })
            append_csv(cfg.csv_log_path, {
                "time": datetime.now().isoformat(),
                "scent": name,
                "ml_match": ml_match,
                "sentiment": sent,
                "final_score": final_score,
                "user": reply,
            })

        # --- Blend selection ---
        state = SessionState.BLEND_SELECT
        ranked = sorted(notes_results, key=lambda x: x["final_score"], reverse=True)
        top2 = ranked[:2]
        score_bits = [f"{x['scent_id']}={x['final_score']:.2f}" for x in top2]
        summary = (
            f"Top blend candidates: {', '.join(x['scent_id'] for x in top2)}. "
            f"Scores: {', '.join(score_bits)}."
        )
        console.print(Panel(summary, title="Results"))
        if client:
            fin = llm_reply(
                client, cfg.zhipu_model, SYSTEM_PROMPT,
                f"Reveal the top 2 scents {top2} and relate briefly to biometrics.",
            )
            console.print(Panel(fin, title="Assistant"))
        append_csv(cfg.csv_log_path, {"time": datetime.now().isoformat(), "summary": summary})
        state = SessionState.COMPLETE
    finally:
        if ard:
            ard.send_command("PUMP:OFF", expect_ack=False)
        if esp:
            esp.close()
        if ard:
            ard.close()
        if mock_feed:
            mock_feed.stop()


class MockSensorFeed:
    """Generate synthetic GSR/HR/HRV for end-to-end tests without hardware."""

    def __init__(self) -> None:
        """Initialise internal random-walk state."""
        self._t = 0.0
        self._gsr = 3.0
        self._hr = 72.0
        self._hrv = 40.0
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self) -> None:
        """No-op thread for API symmetry (generation is synchronous here)."""
        return

    def stop(self) -> None:
        """Stop background activity (unused)."""
        self._stop.set()

    def next_reading(self) -> SensorReading:
        """Produce the next synthetic sensor row (~4 Hz if called at that rate)."""
        self._t += 0.25
        self._gsr += 0.15 * math.sin(self._t * 0.7) + random.uniform(-0.05, 0.05)
        self._gsr = max(0.5, min(20.0, self._gsr))
        self._hr += random.uniform(-0.8, 0.8)
        self._hr += (72.0 - self._hr) * 0.03
        self._hr = max(50.0, min(110.0, self._hr))
        self._hrv = 35.0 + 10.0 * math.sin(self._t * 0.2) + random.uniform(-2, 2)
        self._hrv = max(5.0, min(90.0, self._hrv))
        return SensorReading(gsr=self._gsr, hr=self._hr, hrv=self._hrv)


# === MOCK MODE FOR TESTING WITHOUT HARDWARE ===
# When ``--mock-mode`` is set, no serial ports are opened; :class:`MockSensorFeed`
# feeds synthetic data. Combine with ``ensure_stub_model`` so inference runs
# before you train a deployment model in ``scentsation_ml``.


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the hub."""
    p = argparse.ArgumentParser(description="Scentsation central hub")
    p.add_argument("--esp32-port", default=os.environ.get("ESP32_PORT", "/dev/ttyUSB0"))
    p.add_argument("--arduino-port", default=os.environ.get("ARDUINO_PORT", "/dev/ttyUSB1"))
    p.add_argument("--model-path", default="models/classifier.joblib")
    p.add_argument("--duration-calibration", type=int, default=30)
    p.add_argument("--duration-spray", type=int, default=5)
    p.add_argument("--duration-observe", type=int, default=10)
    p.add_argument(
        "--scent-notes",
        default="Peppermint,Lemon,Rosemary,Ginger",
        help="Comma-separated scent names (order matches pump indices 1..N)",
    )
    p.add_argument("--no-llm", action="store_true", help="Skip Zhipu API calls")
    p.add_argument("--mock-mode", action="store_true", help="Synthetic sensors, no USB serial")
    p.add_argument("--no-dashboard", action="store_true", help="Disable Rich live refresh snippets")
    p.add_argument(
        "--allow-bare-model",
        action="store_true",
        help=(
            "Allow a bare sklearn estimator in joblib (no dict / no feature_names). "
            "Not recommended for deployment. Env: SCENTSATION_ALLOW_BARE_MODEL=1."
        ),
    )
    p.add_argument(
        "--require-hub-feature-json",
        action="store_true",
        help=(
            "Fail if config/hub_feature_names.json is missing (no fallback). "
            "Sets SCENTSATION_REQUIRE_HUB_FEATURE_JSON for this process. "
            "Env alone: SCENTSATION_REQUIRE_HUB_FEATURE_JSON=1."
        ),
    )
    p.add_argument(
        "--demo-strict",
        action="store_true",
        help=(
            "Competition/demo profile: require hub_feature_names.json (strict), "
            "reject bare-model flags and bare joblibs. "
            "Equivalent to SCENTSATION_REQUIRE_HUB_FEATURE_JSON=1 plus extra checks. "
            "Env: SCENTSATION_DEMO_STRICT=1."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Program entry: load config, model, and run one session."""
    global HUB_FEATURE_NAMES
    args = parse_args(argv)
    demo_strict = bool(args.demo_strict) or (
        os.getenv("SCENTSATION_DEMO_STRICT", "").strip() == "1"
    )
    if demo_strict:
        os.environ["SCENTSATION_REQUIRE_HUB_FEATURE_JSON"] = "1"
    if args.require_hub_feature_json:
        os.environ["SCENTSATION_REQUIRE_HUB_FEATURE_JSON"] = "1"
    if demo_strict:
        env_bare = os.getenv("SCENTSATION_ALLOW_BARE_MODEL", "").strip() == "1"
        if args.allow_bare_model or env_bare:
            console.print(
                Panel(
                    "[bold]--demo-strict[/bold] (or SCENTSATION_DEMO_STRICT=1) is incompatible "
                    "with bare-model mode. Remove --allow-bare-model and unset "
                    "SCENTSATION_ALLOW_BARE_MODEL.",
                    title="Configuration error",
                    border_style="red",
                )
            )
            sys.exit(1)
    if os.getenv("SCENTSATION_REQUIRE_HUB_FEATURE_JSON", "").strip() == "1":
        HUB_FEATURE_NAMES = load_hub_feature_names_tuple()

    cfg = Config(
        esp32_port=args.esp32_port,
        arduino_port=args.arduino_port,
        model_path=args.model_path,
        calibration_duration=args.duration_calibration,
        spray_duration=args.duration_spray,
        observe_duration=args.duration_observe,
        scent_notes=[s.strip() for s in args.scent_notes.split(",") if s.strip()],
        use_llm=not args.no_llm,
        mock_mode=args.mock_mode,
        no_dashboard=args.no_dashboard,
        zhipu_api_key=os.getenv("ZHIPU_API_KEY", ""),
    )
    if not cfg.use_llm:
        cfg.zhipu_api_key = ""

    if not cfg.mock_mode:
        logger.info(
            "USB serial: replugging ESP32/Arduino requires restarting this process; "
            "hot-plug recovery is not implemented."
        )
        preflight_serial_ports(cfg)

    ensure_stub_model(cfg.model_path)
    model, scaler, payload_meta = load_model_payload(cfg.model_path)
    if demo_strict and payload_meta is None:
        console.print(
            Panel(
                "[bold]--demo-strict[/bold] requires a dict joblib with "
                "[bold]feature_names[/bold] (not a bare estimator). "
                "Export with scentsation_ml.export.dump_hub_joblib.",
                title="Incompatible joblib",
                border_style="red",
            )
        )
        sys.exit(1)
    allow_bare = args.allow_bare_model or (
        os.getenv("SCENTSATION_ALLOW_BARE_MODEL", "").strip() == "1"
    )
    validate_model_for_hub(
        model,
        scaler,
        HUB_FEATURE_DIM,
        payload=payload_meta,
        allow_bare_model=allow_bare,
    )
    console.print(Panel(str(cfg), title="Config"))
    run_session(cfg, model, scaler)


if __name__ == "__main__":
    main()
