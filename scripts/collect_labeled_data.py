#!/usr/bin/env python3
"""
Record one labeled physiological session for Scentsation ML (Phase 1).

Streams GSR/HR/HRV from the ESP32 at ~4 Hz (same path as scentsation_hub) and writes
``data/raw_labeled/{subject_id}_{label}_{ISO8601}.csv``.

Induction protocols (operator reads aloud before recording):
  NEUTRAL  — Eyes open, seated rest, no task, no phone.
  RELAXED  — 4-7-8 breathing or short guided meditation.
  STRESSED — Mental arithmetic under time pressure (e.g. count down from 1001 by 7s).
  FOCUSED  — Sustained attention task (Stroop, N-back, or real work the subject cares about).

Randomize label order across subjects to reduce order effects.

Mock mode (--mock-mode) exercises the pipeline only; synthetic data is not class-specific.

Phase 5 (booth / venue calibration)
------------------------------------
Before a demo, record 2–3 short sessions on the **same machine and room** as the
conference (HVAC and lighting shift GSR baselines). Use a dedicated ``subject_id``
(e.g. ``venue_cal_2026_04_17``). After saving CSVs under ``data/raw_labeled/``,
run ``python -m scentsation_ml.build_custom_6d`` and ``python scentsation_ml/train_hub_svm.py``
(or ``scripts/venue_retrain.sh``) so ``datasets/custom_6d.csv`` and
``models/classifier.joblib`` include that environment.

Run from repo root (use ``/dev/cu.*`` on macOS). Arduino Uno with hub-format firmware
(``GSR:,HR:,HRV:`` lines) is fine — ``--sensor-port`` is the same as ``--esp32-port``::

  python3 scripts/collect_labeled_data.py --subject-id S01 --label FOCUSED --duration 180 \\
      --sensor-port /dev/cu.usbserial-XXXX
  python3 scripts/collect_labeled_data.py --subject-id S01 --label NEUTRAL --duration 60 --mock-mode
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

# Repo root for `import scentsation_hub`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scentsation_hub import (  # noqa: E402
    CLASS_ORDER,
    Config,
    MockSensorFeed,
    SensorReading,
    SerialManager,
)

try:
    import serial
except ImportError:
    serial = None  # type: ignore


def _preflight_esp32_only(port: str, baud: int = 115200) -> None:
    """Open only the ESP32 port (collector does not use the Arduino)."""
    if serial is None:
        raise RuntimeError("pyserial is not installed")
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
        ser.close()
    except Exception as e:
        raise RuntimeError(f"Could not open ESP32 port {port!r}: {e}") from e

PROTOCOLS = {
    "NEUTRAL": "Eyes open, seated rest, no task, no phone.",
    "RELAXED": "4-7-8 breathing or short guided meditation.",
    "STRESSED": "Mental arithmetic under time pressure (e.g. count down from 1001 by 7s).",
    "FOCUSED": "Sustained attention task (Stroop, N-back, or real work).",
}


def _default_output_path(output_dir: Path, subject_id: str, label: str, session_id: str) -> Path:
    safe_label = label.strip().upper().replace(" ", "_")
    return output_dir / f"{subject_id}_{safe_label}_{session_id}.csv"


def collect_loop(
    duration_sec: float,
    esp: SerialManager | None,
    mock_feed: MockSensorFeed | None,
    min_samples: int,
) -> list[SensorReading]:
    """Poll sensor queue until ``duration_sec`` elapsed or interrupted."""
    readings: list[SensorReading] = []
    t_end = time.time() + duration_sec
    last_sig: tuple[float, float, float, float] | None = None

    try:
        while time.time() < t_end:
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
                readings.append(r)
                # Mock feed is synchronous — pace at ~4 Hz to match ESP32 serial cadence.
                if mock_feed is not None:
                    time.sleep(0.25)

            elapsed = duration_sec - max(0.0, t_end - time.time())
            if readings:
                last = readings[-1]
                print(
                    f"\r  {elapsed:5.1f}s / {duration_sec:.0f}s  "
                    f"GSR={last.gsr:5.2f}  HR={last.hr:5.1f}  HRV={last.hrv:5.1f}  n={len(readings)}",
                    end="",
                    flush=True,
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped early.")

    print()
    return readings


def main() -> None:
    p = argparse.ArgumentParser(description="Collect one labeled ESP32 sensor session")
    p.add_argument("--subject-id", required=True, help="Subject identifier, e.g. S01")
    p.add_argument(
        "--label",
        required=True,
        choices=CLASS_ORDER,
        help="Induced physiological state label",
    )
    p.add_argument("--duration", type=float, default=180.0, help="Recording length in seconds")
    p.add_argument(
        "--esp32-port",
        "--sensor-port",
        default=os.environ.get("ESP32_PORT", "/dev/ttyUSB0"),
        dest="esp32_port",
        help="USB serial for GSR:,HR:,HRV: stream (Uno/ESP32/etc.). Env: ESP32_PORT.",
    )
    p.add_argument("--mock-mode", action="store_true", help="Synthetic data; no USB (pipeline test only)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "data" / "raw_labeled",
        help="Directory for CSV output",
    )
    args = p.parse_args()

    min_samples = max(1, int(0.5 * args.duration * 4.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    session_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = _default_output_path(args.output_dir, args.subject_id, args.label, session_id)
    if out_path.exists():
        print(f"Refusing to overwrite existing file: {out_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Label: {args.label}")
    print(f"Protocol: {PROTOCOLS[args.label]}")
    print("Press Enter when the subject is ready to begin recording…", end=" ")
    input()

    esp: SerialManager | None = None
    mock_feed: MockSensorFeed | None = None

    if args.mock_mode:
        mock_feed = MockSensorFeed()
        mock_feed.start()
    else:
        if serial is None:
            print("pyserial is not installed. Use --mock-mode or: pip install pyserial", file=sys.stderr)
            sys.exit(1)
        cfg = Config(
            esp32_port=args.esp32_port,
            arduino_port=os.environ.get("ARDUINO_PORT", "/dev/ttyUSB1"),
            mock_mode=False,
        )
        _preflight_esp32_only(cfg.esp32_port, cfg.baud_rate)
        esp = SerialManager(cfg, "input")
        esp.start()

    print(f"Recording for {args.duration:.0f} seconds… (Ctrl+C to stop early)")
    readings = collect_loop(args.duration, esp, mock_feed, min_samples)

    if esp:
        esp.close()
    if mock_feed:
        mock_feed.stop()

    if len(readings) < min_samples:
        print(
            f"Too few samples ({len(readings)} < minimum {min_samples}). "
            "Check ESP32 cable, baud 115200, and GSR:…,HR:…,HRV:… lines.",
            file=sys.stderr,
        )
        sys.exit(1)

    fieldnames = ("time", "subject_id", "label", "session_id", "gsr", "hr", "hrv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in readings:
            w.writerow(
                {
                    "time": datetime.fromtimestamp(r.timestamp).isoformat(timespec="milliseconds"),
                    "subject_id": args.subject_id,
                    "label": args.label,
                    "session_id": session_id,
                    "gsr": r.gsr,
                    "hr": r.hr,
                    "hrv": r.hrv,
                }
            )

    print(f"Wrote {len(readings)} rows → {out_path}")


if __name__ == "__main__":
    main()
