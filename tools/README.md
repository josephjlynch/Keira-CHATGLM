# `tools/` (optional)

Use this folder for small utilities (serial port listing, log conversion, etc.) that are not part of the main `scentsation_ml` package or the root `scentsation_hub.py` script.

## Hub 6-D classifier (demo deployment)

Canonical feature order is defined once in **[`config/hub_feature_names.json`](../config/hub_feature_names.json)**. The hub and `scentsation_ml.export` both read this file.

- Joblib bundles must be a **dict** with `model`, optional `scaler`, and **`feature_names`** matching that JSON — otherwise `scentsation_hub.py` exits at startup.
- **Competition / production demos:** do **not** use **`--allow-bare-model`** or **`SCENTSATION_ALLOW_BARE_MODEL=1`** — they bypass `feature_names` metadata.
- For legacy single-estimator `.joblib` files (no dict metadata), use **`--allow-bare-model`** or **`SCENTSATION_ALLOW_BARE_MODEL=1`** only in local dev.
- **Strict JSON (recommended for demos / CI):** require the file on disk with no silent fallback — set **`SCENTSATION_REQUIRE_HUB_FEATURE_JSON=1`** before launch, or pass **`--require-hub-feature-json`** to `scentsation_hub.py` (missing `config/hub_feature_names.json` then fails fast).
- **Competition / L’Oréal-style runs:** pass **`--demo-strict`** (or set **`SCENTSATION_DEMO_STRICT=1`**) to enable strict JSON **and** block bare-model flags and bare joblibs in one step — do not combine with **`--allow-bare-model`** or **`SCENTSATION_ALLOW_BARE_MODEL=1`**.
- Drift check: from the repo root run **`make verify-contract`** (runs `python3 scripts/verify_hub_feature_contract.py`) to confirm the JSON matches the built-in fallback tuples in the hub and `scentsation_ml.export`.
- Do **not** copy the 15-D `best_model.joblib` from `scentsation_ml` training and expect correct labels on hardware.
- Use `scentsation_ml.export.build_hub_joblib_payload` / `dump_hub_joblib` when saving a model trained on hub-aligned data.

## USB serial

The hub **attempts to reopen** serial ports after sustained read errors (e.g. brief USB glitches). If a device stays missing or the wrong `ESP32_PORT` / `ARDUINO_PORT` is set, **restart the process** after fixing wiring — recovery is best-effort, not guaranteed.

## Mock mode (`--mock-mode`)

Synthetic data are **demonstration-only** (not clinical): HR/HRV are **coupled** via a simulated RR-interval process; GSR uses tonic drift plus small phasic bumps. For repeatable demos or screenshots, set **`SCENTSATION_MOCK_SEED=123`** (any integer).

## CLI hints

- `scentsation_hub.py --help` documents `--allow-bare-model` and ports. After replug, confirm `ESP32_PORT` / `ARDUINO_PORT` match `ls` of your OS.

## Arduino demo safety

Do **not** send `PUMP:TEST` during a live demo: it runs multi-second `delay()` calls and stalls serial I/O and auto-shutoff. Bench use only, or build firmware with `ALLOW_PUMP_TEST` defined (see `arduino_pumps.ino`).
