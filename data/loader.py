"""
data/loader.py
──────────────
Unified data-loading interface.  Downstream code (preprocessing, model,
dashboard) always calls `load_trace(cfg)` and receives the same DataFrame
schema regardless of whether the source is synthetic or a real CSV file.

Schema guarantee
────────────────
  timestamp : datetime64 (DatetimeIndex after set_index)
  cpu_util  : float, values in [0, 1]

Swapping a real trace in
────────────────────────
Change config.yaml:
    data:
      source: "csv"
      csv_path: "path/to/azure_trace.csv"

The CSV must have:
  • A column named `timestamp` (any format parseable by pd.to_datetime)
  • A column named `cpu_util` (float; values 0–100 are auto-normalised to 0–1)

No other code changes are required.
"""

from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_synthetic(cfg: dict) -> pd.DataFrame:
    from data.synthetic_generator import generate_trace

    data_cfg = cfg["data"]
    return generate_trace(
        n_days=data_cfg.get("n_days", 30),
        freq=data_cfg.get("freq", "1min"),
        seed=data_cfg.get("seed", 42),
    )


def _load_csv(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    path = data_cfg.get("csv_path")
    if not path:
        raise ValueError(
            "config.yaml: data.csv_path must be set when data.source='csv'."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Normalise cpu_util from 0-100 -> 0-1 if needed
    if df["cpu_util"].max() > 1.5:
        df["cpu_util"] = df["cpu_util"] / 100.0

    df["cpu_util"] = df["cpu_util"].clip(0.0, 1.0).astype("float32")
    return df[["timestamp", "cpu_util"]]


# ── public API ────────────────────────────────────────────────────────────────

def load_trace(
    config_path: str | Path = "config.yaml",
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Load the workload trace according to config.yaml.

    Parameters
    ----------
    config_path:
        Path to the YAML config file (used when `cfg` is None).
    cfg:
        Pre-loaded config dict (skips file I/O if already read).

    Returns
    -------
    pd.DataFrame
        Indexed by `timestamp` (DatetimeIndex), single column `cpu_util`.
    """
    if cfg is None:
        cfg = _load_config(config_path)

    source = cfg["data"].get("source", "synthetic")

    if source == "synthetic":
        df = _load_synthetic(cfg)
    elif source == "csv":
        df = _load_csv(cfg)
    else:
        raise ValueError(f"Unknown data.source '{source}'. Use 'synthetic' or 'csv'.")

    # Ensure DatetimeIndex
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    # Resample to the configured frequency (fills any gaps via forward-fill)
    freq = cfg["data"].get("freq", "1min")
    df = df.resample(freq).mean().ffill()

    return df


# ── CLI sanity check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow running from any directory by resolving config relative to project root
    project_root = Path(__file__).parent.parent
    cfg_path = project_root / "config.yaml"

    print(f"Loading trace from: {cfg_path}")
    df = load_trace(config_path=cfg_path)
    print(f"\nTrace shape : {df.shape}")
    print(f"Date range  : {df.index[0]}  ->  {df.index[-1]}")
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string())
    print(f"\nStats:\n{df['cpu_util'].describe()}")
