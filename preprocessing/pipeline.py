"""
preprocessing/pipeline.py
──────────────────────────
Three sequential steps applied to the raw workload trace before it is fed to
the LSTM:

  1. resample   — forward-fill to a uniform time grid (already done in loader,
                  but the function is exposed here for completeness / testing).
  2. normalize  — MinMaxScaler -> [0, 1]; the scaler is stored so we can
                  inverse-transform model outputs back to the original scale.
  3. make_windows — sliding-window partitioning into (X, y) pairs suitable
                    for sequence-to-sequence supervised learning.

Public API
──────────
  PreprocessingPipeline(cfg)
      .fit_transform(df)  -> (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
      .inverse_transform(arr) -> np.ndarray in original scale

  Standalone helpers:
      resample(df, freq)
      normalize(df)  -> (scaled_series, scaler)
      make_windows(series, window, horizon) -> (X, y)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── standalone helpers ────────────────────────────────────────────────────────

def resample(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Resample `df` to `freq` and forward-fill gaps."""
    return df.resample(freq).mean().ffill()


def normalize(series: np.ndarray | pd.Series) -> tuple[np.ndarray, MinMaxScaler]:
    """Fit a MinMaxScaler on `series` and return (scaled_array, scaler).

    Parameters
    ----------
    series : 1-D array-like

    Returns
    -------
    scaled : np.ndarray, shape (N,), dtype float32, values in [0, 1]
    scaler : fitted MinMaxScaler (use .inverse_transform to undo)
    """
    arr = np.asarray(series, dtype=np.float32).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(arr).ravel().astype(np.float32)
    return scaled, scaler


def make_windows(
    series: np.ndarray,
    window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window (X, y) pairs.

    Parameters
    ----------
    series  : 1-D float array of length N.
    window  : number of past timesteps fed as input (look-back).
    horizon : number of future timesteps to predict.

    Returns
    -------
    X : np.ndarray, shape (num_samples, window, 1)  — LSTM input
    y : np.ndarray, shape (num_samples, horizon)     — targets
    """
    X, y = [], []
    max_start = len(series) - window - horizon + 1
    for i in range(max_start):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + horizon])
    X = np.array(X, dtype=np.float32)[:, :, np.newaxis]  # (N, W, 1)
    y = np.array(y, dtype=np.float32)                     # (N, H)
    return X, y


# ── pipeline class ────────────────────────────────────────────────────────────

class PreprocessingPipeline:
    """Chain resample -> normalize -> sliding windows, then split train/val/test.

    Usage
    -----
    >>> pipeline = PreprocessingPipeline(cfg)
    >>> splits, scaler = pipeline.fit_transform(df)
    >>> X_train, y_train, X_val, y_val, X_test, y_test = splits
    """

    def __init__(self, cfg: dict):
        self.freq       = cfg["data"]["freq"]
        self.window     = cfg["preprocessing"]["window_size"]
        self.horizon    = cfg["preprocessing"]["horizon"]
        self.train_frac = cfg["preprocessing"]["train_frac"]
        self.val_frac   = cfg["preprocessing"]["val_frac"]
        self.scaler: MinMaxScaler | None = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _split(self, X: np.ndarray, y: np.ndarray):
        """Chronological (non-shuffled) train / val / test split."""
        n = len(X)
        train_end = int(n * self.train_frac)
        val_end   = train_end + int(n * self.val_frac)
        return (
            X[:train_end],  y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:],    y[val_end:],
        )

    # ── public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[tuple[np.ndarray, ...], MinMaxScaler]:
        """Run the full pipeline on a raw trace DataFrame.

        Parameters
        ----------
        df : DataFrame with a `cpu_util` column (DatetimeIndex).

        Returns
        -------
        splits : (X_train, y_train, X_val, y_val, X_test, y_test)
        scaler : fitted MinMaxScaler
        """
        # 1. Resample (idempotent if already done in loader)
        df = resample(df, self.freq)

        # 2. Normalize
        scaled, self.scaler = normalize(df["cpu_util"].values)

        # 3. Sliding windows
        X, y = make_windows(scaled, self.window, self.horizon)

        # 4. Split
        splits = self._split(X, y)
        return splits, self.scaler

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled values back to original [0, 1] cpu_util range.

        Parameters
        ----------
        arr : np.ndarray, any shape — will be flattened, transformed, reshaped.
        """
        if self.scaler is None:
            raise RuntimeError("Call fit_transform before inverse_transform.")
        orig_shape = arr.shape
        flat = arr.reshape(-1, 1)
        return self.scaler.inverse_transform(flat).reshape(orig_shape)


# ── CLI sanity check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from data.loader import load_trace

    project_root = Path(__file__).parent.parent
    cfg_path = project_root / "config.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading trace …")
    df = load_trace(cfg=cfg)
    print(f"  Raw trace shape: {df.shape}")

    print("\nRunning PreprocessingPipeline …")
    pipeline = PreprocessingPipeline(cfg)
    splits, scaler = pipeline.fit_transform(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits

    print(f"  X_train : {X_tr.shape}   y_train : {y_tr.shape}")
    print(f"  X_val   : {X_val.shape}   y_val   : {y_val.shape}")
    print(f"  X_test  : {X_te.shape}    y_test  : {y_te.shape}")
    print(f"\n  Scaler range: {scaler.data_min_[0]:.4f} -> {scaler.data_max_[0]:.4f}")
    print("\n[OK] Preprocessing pipeline OK")
