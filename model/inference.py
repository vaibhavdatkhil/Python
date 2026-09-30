"""
model/inference.py
──────────────────
MC Dropout inference — runs N stochastic forward passes and returns a
point forecast (mean) plus a confidence interval (percentile-based).

Public API
──────────
  load_checkpoint(ckpt_path, device) -> (model, scaler, cfg)
  mc_predict(model, x_window, n_samples, scaler, lower_pct, upper_pct)
      -> (mean_forecast, lower_bound, upper_bound)   [original scale]

CLI usage
─────────
  python -m model.inference
  -> prints a sample prediction for the first test window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import yaml

from model.lstm_model import LSTMForecast, build_model, mc_dropout_ctx


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[LSTMForecast, MinMaxScaler, dict]:
    """Load a saved checkpoint and reconstruct the model + scaler.

    Parameters
    ----------
    ckpt_path : path to `best_model.pt`.
    device    : torch device.

    Returns
    -------
    model  : LSTMForecast in eval mode.
    scaler : MinMaxScaler fitted on the training data.
    cfg    : the config dict that was used to train.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Reconstruct scaler from saved min/max
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_  = np.array(ckpt["scaler_min"], dtype=np.float32)
    scaler.data_max_  = np.array(ckpt["scaler_max"], dtype=np.float32)
    scaler.scale_     = 1.0 / (scaler.data_max_ - scaler.data_min_ + 1e-8)
    scaler.data_range_= scaler.data_max_ - scaler.data_min_
    scaler.min_       = -scaler.scale_ * scaler.data_min_
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = 1  # dummy; needed for sklearn ≥ 1.0

    return model, scaler, cfg


# ── MC inference ──────────────────────────────────────────────────────────────

def mc_predict(
    model:     LSTMForecast,
    x_window:  np.ndarray,
    n_samples: int = 50,
    scaler:    MinMaxScaler | None = None,
    lower_pct: int = 5,
    upper_pct: int = 95,
    device:    torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MC Dropout inference and return mean + CI bounds.

    Parameters
    ----------
    model     : trained LSTMForecast.
    x_window  : np.ndarray, shape (window, 1) or (1, window, 1) — one sample.
    n_samples : number of stochastic forward passes.
    scaler    : if provided, inverse-transforms output to original scale.
    lower_pct : lower percentile for CI (e.g. 5  -> 5th percentile).
    upper_pct : upper percentile for CI (e.g. 95 -> 95th percentile).
    device    : torch device.

    Returns
    -------
    mean  : np.ndarray, shape (horizon,)  — point forecast.
    lower : np.ndarray, shape (horizon,)  — lower CI bound.
    upper : np.ndarray, shape (horizon,)  — upper CI bound.
    """
    # Ensure shape (1, window, 1)
    x = np.asarray(x_window, dtype=np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, :, np.newaxis]   # (1, W, 1)
    elif x.ndim == 2:
        x = x[np.newaxis]                  # (1, W, 1)

    x_tensor = torch.from_numpy(x).expand(n_samples, -1, -1).contiguous().to(device)

    # Stochastic forward passes (batched parallel)
    with mc_dropout_ctx(model):
        with torch.no_grad():
            out = model(x_tensor)               # (n_samples, horizon)
            preds = out.cpu().numpy()

    mean  = preds.mean(axis=0)                  # (horizon,)
    lower = np.percentile(preds, lower_pct, axis=0)
    upper = np.percentile(preds, upper_pct, axis=0)

    # Inverse-transform if scaler is provided
    if scaler is not None:
        def _inv(arr):
            return scaler.inverse_transform(arr.reshape(-1, 1)).ravel()
        mean  = _inv(mean)
        lower = _inv(lower)
        upper = _inv(upper)

    return mean, lower, upper


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from data.loader import load_trace
    from preprocessing.pipeline import PreprocessingPipeline

    project_root = Path(__file__).parent.parent
    cfg_path     = project_root / "config.yaml"
    ckpt_path    = project_root / "checkpoints" / "best_model.pt"

    if not ckpt_path.exists():
        print(f"[X]  Checkpoint not found at {ckpt_path}.")
        print("   Run `python -m model.train` first.")
        raise SystemExit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, _ = load_checkpoint(ckpt_path, device=device)

    # Load test data to grab a sample window
    df = load_trace(cfg=cfg)
    pipeline = PreprocessingPipeline(cfg)
    splits, _ = pipeline.fit_transform(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits

    inf_cfg   = cfg["inference"]
    n_samples = inf_cfg["n_mc_samples"]
    lo        = inf_cfg["ci_lower_pct"]
    hi        = inf_cfg["ci_upper_pct"]

    sample_window = X_te[0]   # first test window
    mean, lower, upper = mc_predict(
        model, sample_window, n_samples=n_samples,
        scaler=scaler, lower_pct=lo, upper_pct=hi, device=device,
    )

    print(f"MC Dropout inference ({n_samples} samples, CI {lo}th-{hi}th pct):")
    print(f"  Horizon   : {len(mean)} steps")
    print(f"  Mean      : {mean}")
    print(f"  Lower     : {lower}")
    print(f"  Upper     : {upper}")
    print("\n[OK] Inference OK")
