#!/usr/bin/env python3
"""
Live drift check (Phase 4): stream sensors through the same path as the hub
(rolling buffer → ``compute_features`` → joblib model) for a fixed duration.

**Protocol:** sit ~2 minutes in a neutral resting state. Predictions should
mostly stay **NEUTRAL** or **RELAXED**. If labels flicker across all four states
at high rate, widen ``--buffer-window-sec`` (matches hub ``buffer_window_sec``)
or increase ``--vote-window-sec`` for a calmer rolling majority readout.

Run from repo root::

    python scripts/live_drift_check.py --model-path models/classifier.joblib \\
        --duration-sec 120 --mock-mode
    python scripts/live_drift_check.py --model-path models/classifier.joblib \\
        --duration-sec 120 --esp32-port /dev/tty.usbserial-XXXX

Requires pyserial for non-mock mode.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scentsation_hub import (  # noqa: E402
    HUB_FEATURE_DIM,
    Config,
    MockSensorFeed,
    RollingBuffer,
    SensorReading,
    SerialManager,
    apply_model,
    compute_features,
    ensure_stub_model,
    load_model_payload,
    validate_model_for_hub,
)

try:
    import serial
except ImportError:
    serial = None  # type: ignore


def _preflight_esp32_only(port: str, baud: int = 115200) -> None:
    if serial is None:
        raise RuntimeError("pyserial is not installed")
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
        ser.close()
    except Exception as e:
        raise RuntimeError(f"Could not open ESP32 port {port!r}: {e}") from e


def main() -> None:
    p = argparse.ArgumentParser(description="Neutral-state live ML drift check (~hub inference path)")
    p.add_argument("--model-path", default="models/classifier.joblib")
    p.add_argument("--duration-sec", type=float, default=120.0)
    p.add_argument("--buffer-window-sec", type=float, default=10.0, help="Rolling buffer length (hub default 10)")
    p.add_argument(
        "--vote-window-sec",
        type=float,
        default=10.0,
        help="Wall-clock window for rolling majority vote over recent predictions",
    )
    p.add_argument("--log-interval-sec", type=float, default=1.0, help="Print one line at most this often")
    p.add_argument(
        "--esp32-port",
        "--sensor-port",
        default=os.environ.get("ESP32_PORT", "/dev/ttyUSB0"),
        dest="esp32_port",
        help="USB serial for GSR:,HR:,HRV: stream",
    )
    p.add_argument("--mock-mode", action="store_true")
    p.add_argument(
        "--allow-bare-model",
        action="store_true",
        help="Same as hub: allow bare sklearn joblib (not recommended).",
    )
    args = p.parse_args()

    model_path = args.model_path
    if not Path(model_path).is_file():
        ensure_stub_model(model_path)

    model, scaler, payload = load_model_payload(model_path)
    validate_model_for_hub(
        model,
        scaler,
        HUB_FEATURE_DIM,
        payload=payload,
        allow_bare_model=args.allow_bare_model,
    )

    buf = RollingBuffer(args.buffer_window_sec)
    pred_history: Deque[Tuple[float, str]] = deque()
    esp: SerialManager | None = None
    mock_feed: MockSensorFeed | None = None
    last_sig: tuple[float, float, float, float] | None = None

    if args.mock_mode:
        mock_feed = MockSensorFeed()
        mock_feed.start()
    else:
        if serial is None:
            print("Install pyserial or use --mock-mode.", file=sys.stderr)
            sys.exit(1)
        cfg = Config(
            esp32_port=args.esp32_port,
            arduino_port=os.environ.get("ARDUINO_PORT", "/dev/ttyUSB1"),
            mock_mode=False,
        )
        _preflight_esp32_only(cfg.esp32_port, cfg.baud_rate)
        esp = SerialManager(cfg, "input")
        esp.start()

    t_end = time.time() + args.duration_sec
    last_log = 0.0
    print(
        f"Drift check: {args.duration_sec:.0f}s  buffer={args.buffer_window_sec}s  "
        f"vote_window={args.vote_window_sec}s  log_every={args.log_interval_sec}s"
    )
    print("columns: time  pred  majority_last_window  dist_in_window")

    try:
        while time.time() < t_end:
            now = time.time()
            r: SensorReading | None = None
            if mock_feed is not None:
                r = mock_feed.next_reading()
            elif esp is not None:
                try:
                    r = esp.queue.get(timeout=0.2)
                except queue.Empty:
                    lr = esp.get_latest()
                    if lr is not None:
                        sig = (lr.timestamp, lr.gsr, lr.hr, lr.hrv)
                        if last_sig != sig:
                            r = lr
                            last_sig = sig

            if r is not None:
                buf.push(r)
                feats = compute_features(buf.items)
                pred = apply_model(model, scaler, feats)
                pred_history.append((now, pred))
                cutoff = now - args.vote_window_sec
                while pred_history and pred_history[0][0] < cutoff:
                    pred_history.popleft()

            if now - last_log >= args.log_interval_sec:
                last_log = now
                window_preds = [pr for _, pr in pred_history]
                maj = Counter(window_preds).most_common(1)[0][0] if window_preds else "—"
                dist = dict(Counter(window_preds)) if window_preds else {}
                last_p = window_preds[-1] if window_preds else "—"
                ts = datetime.now().isoformat(timespec="seconds")
                print(f"{ts}  {last_p:<8}  {maj:<8}  {dist}")

            if mock_feed is not None:
                time.sleep(0.25)
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        if esp:
            esp.close()
        if mock_feed:
            mock_feed.stop()

    print("Done.")


if __name__ == "__main__":
    main()
