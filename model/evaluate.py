"""
model/evaluate.py
──────────────────
Loads the best checkpoint, runs MC Dropout inference on the test split, and:

  1. Computes  MAE, RMSE  (sklearn)
  2. Computes  interval coverage  (% of actual points inside [lower, upper])
  3. Saves a   forecast-vs-actual + shaded CI band plot  to outputs/

Usage
─────
  python -m model.evaluate
  python -m model.evaluate --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch

from data.loader import load_trace
from model.inference import load_checkpoint, mc_predict
from preprocessing.pipeline import PreprocessingPipeline


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> dict:
    """Compute MAE, RMSE, and interval coverage.

    Parameters
    ----------
    actuals   : shape (N, horizon) — ground-truth values.
    forecasts : shape (N, horizon) — mean predictions.
    lowers    : shape (N, horizon) — lower CI bounds.
    uppers    : shape (N, horizon) — upper CI bounds.

    Returns
    -------
    dict with keys: mae, rmse, coverage
    """
    flat_actual   = actuals.ravel()
    flat_forecast = forecasts.ravel()
    flat_lower    = lowers.ravel()
    flat_upper    = uppers.ravel()

    mae  = mean_absolute_error(flat_actual, flat_forecast)
    rmse = np.sqrt(mean_squared_error(flat_actual, flat_forecast))

    # Interval coverage: fraction of actuals inside [lower, upper]
    in_interval = np.logical_and(flat_actual >= flat_lower, flat_actual <= flat_upper)
    coverage    = float(in_interval.mean())

    return {"mae": mae, "rmse": rmse, "coverage": coverage}


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_forecast(
    actuals:   np.ndarray,
    forecasts: np.ndarray,
    lowers:    np.ndarray,
    uppers:    np.ndarray,
    metrics:   dict,
    output_path: Path,
    n_display: int = 500,
) -> None:
    """Save a forecast-vs-actual plot with shaded CI band.

    Only plots the first `n_display` horizon-step flattened values for clarity.
    """
    # Flatten in row-major order (sample-by-sample multi-step sequence)
    actual_flat   = actuals[:n_display].ravel()
    forecast_flat = forecasts[:n_display].ravel()
    lower_flat    = lowers[:n_display].ravel()
    upper_flat    = uppers[:n_display].ravel()
    x             = np.arange(len(actual_flat))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(
        "LSTM + MC Dropout — Forecast vs Actual\n"
        f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"Coverage={metrics['coverage']*100:.1f}%",
        fontsize=13,
        fontweight="bold",
    )

    # ── top panel: full series + CI band ──────────────────────────────────────
    ax = axes[0]
    ax.plot(x, actual_flat, color="#2563EB", linewidth=0.8,
            label="Actual", alpha=0.9)
    ax.plot(x, forecast_flat, color="#F59E0B", linewidth=0.8,
            label="Forecast (mean)", alpha=0.9)
    ax.fill_between(x, lower_flat, upper_flat,
                    color="#F59E0B", alpha=0.18, label="CI band (5–95th pct)")
    ax.set_xlabel("Time step (flattened)", fontsize=10)
    ax.set_ylabel("CPU Utilisation", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── bottom panel: residuals ───────────────────────────────────────────────
    ax2 = axes[1]
    residuals = actual_flat - forecast_flat
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.plot(x, residuals, color="#10B981", linewidth=0.5, alpha=0.7)
    ax2.fill_between(x, residuals, 0,
                     where=residuals >= 0, color="#10B981", alpha=0.2, label="Over-estimate (actual > pred)")
    ax2.fill_between(x, residuals, 0,
                     where=residuals < 0,  color="#EF4444", alpha=0.2, label="Under-estimate (actual < pred)")
    ax2.set_xlabel("Time step (flattened)", fontsize=10)
    ax2.set_ylabel("Residual (actual − forecast)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {output_path.resolve()}")


# ── main evaluation function ──────────────────────────────────────────────────

def evaluate(cfg: dict, verbose: bool = True) -> dict:
    """Run full evaluation on the test split.

    Returns
    -------
    dict with keys: mae, rmse, coverage, and raw arrays.
    """
    project_root = Path(__file__).parent.parent
    ckpt_path    = project_root / cfg["training"]["checkpoint_dir"] / "best_model.pt"
    output_dir   = project_root / cfg["evaluation"]["output_dir"]

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run `python -m model.train` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device    : {device}")
        print(f"Checkpoint: {ckpt_path}")

    # Load model + scaler
    model, scaler, _ = load_checkpoint(ckpt_path, device=device)

    # Rebuild test split
    if verbose:
        print("Loading data and rebuilding test split …")
    df = load_trace(cfg=cfg)
    pipeline = PreprocessingPipeline(cfg)
    splits, _ = pipeline.fit_transform(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits

    if verbose:
        print(f"  Test samples : {len(X_te)}")

    # MC inference on all test windows
    inf_cfg   = cfg["inference"]
    n_samples = inf_cfg["n_mc_samples"]
    lo_pct    = inf_cfg["ci_lower_pct"]
    hi_pct    = inf_cfg["ci_upper_pct"]

    forecasts_list, lowers_list, uppers_list = [], [], []

    if verbose:
        print(f"Running MC inference ({n_samples} samples per window) …")

    for i, x_win in enumerate(X_te):
        mean, lower, upper = mc_predict(
            model, x_win,
            n_samples=n_samples, scaler=scaler,
            lower_pct=lo_pct, upper_pct=hi_pct,
            device=device,
        )
        forecasts_list.append(mean)
        lowers_list.append(lower)
        uppers_list.append(upper)

        if verbose and (i + 1) % 500 == 0:
            print(f"  … {i+1}/{len(X_te)}")

    forecasts = np.array(forecasts_list)   # (N_test, horizon)
    lowers    = np.array(lowers_list)
    uppers    = np.array(uppers_list)

    # Inverse-transform actuals for fair comparison
    actuals = scaler.inverse_transform(y_te.reshape(-1, 1)).reshape(y_te.shape)

    # Metrics
    metrics = compute_metrics(actuals, forecasts, lowers, uppers)

    if verbose:
        print("\n-- Evaluation Results ----------------------------------")
        print(f"  MAE              : {metrics['mae']:.5f}")
        print(f"  RMSE             : {metrics['rmse']:.5f}")
        print(f"  Interval coverage: {metrics['coverage']*100:.1f}%  "
              f"(nominal: {hi_pct - lo_pct}%)")
        print("--------------------------------------------------------")

    # Plot
    plot_path = output_dir / "evaluation_plot.png"
    plot_forecast(actuals, forecasts, lowers, uppers, metrics, plot_path)

    return {**metrics, "forecasts": forecasts, "actuals": actuals,
            "lowers": lowers, "uppers": uppers}


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the LSTM forecaster.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    cfg_path     = project_root / args.config

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, verbose=True)
