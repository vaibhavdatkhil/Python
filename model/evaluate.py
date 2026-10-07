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
import pandas as pd
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

def evaluate_windows(
    model,
    scaler,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    device: torch.device,
    label: str,
    output_plot: Path,
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run MC Dropout evaluation on a set of (X, y) windows and save plot."""
    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    inf_cfg   = cfg["inference"]
    n_samples = inf_cfg["n_mc_samples"]
    lo_pct    = inf_cfg["ci_lower_pct"]
    hi_pct    = inf_cfg["ci_upper_pct"]

    forecasts_list, lowers_list, uppers_list = [], [], []

    if verbose:
        print(f"\nRunning MC inference on {label} ({len(X)} windows, {n_samples} samples per window) …")

    for i, x_win in enumerate(X):
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
            print(f"  … {i+1}/{len(X)}")

    forecasts = np.array(forecasts_list)
    lowers    = np.array(lowers_list)
    uppers    = np.array(uppers_list)

    # Inverse-transform actuals for fair comparison
    actuals = scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
    metrics = compute_metrics(actuals, forecasts, lowers, uppers)

    if verbose:
        print(f"\n-- Results: {label} ----------------------------------")
        print(f"  MAE              : {metrics['mae']:.5f}")
        print(f"  RMSE             : {metrics['rmse']:.5f}")
        print(f"  Interval coverage: {metrics['coverage']*100:.1f}%  (nominal: {hi_pct - lo_pct}%)")
        print("--------------------------------------------------------")

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plot_forecast(actuals, forecasts, lowers, uppers, metrics, output_plot)

    return {**metrics, "forecasts": forecasts, "actuals": actuals,
            "lowers": lowers, "uppers": uppers}


def evaluate(
    cfg: dict,
    benchmark: str = "none",
    eval_trace: str | None = None,
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run evaluation on the test split and optional benchmark traces.

    Returns
    -------
    dict with metrics for the primary test split.
    """
    project_root = Path(__file__).parent.parent
    ckpt_path    = project_root / cfg["training"]["checkpoint_dir"] / "best_model.pt"
    output_dir   = project_root / cfg["evaluation"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

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

    primary_res = evaluate_windows(
        model, scaler, X_te, y_te, cfg, device,
        label="Held-out Test Split",
        output_plot=output_dir / "evaluation_plot.png",
        max_samples=max_samples,
        verbose=verbose,
    )

    # Benchmark evaluations if requested
    benchmarks_to_run = []
    if eval_trace:
        benchmarks_to_run.append(("Custom Trace", Path(eval_trace), "eval_custom.png"))
    if benchmark in ("azure", "all"):
        benchmarks_to_run.append(("Azure VM Trace", project_root / "data" / "azure_vm_workload_trace.csv", "eval_azure.png"))
    if benchmark in ("alibaba", "all"):
        benchmarks_to_run.append(("Alibaba Cluster Trace", project_root / "data" / "alibaba_cluster_trace.csv", "eval_alibaba.png"))

    from preprocessing.pipeline import make_windows
    W = cfg["preprocessing"]["window_size"]
    H = cfg["preprocessing"]["horizon"]

    all_results = {"test_split": primary_res}

    for label, path, plot_file in benchmarks_to_run:
        if not path.exists():
            print(f"\n[WARN] Benchmark trace {path} not found, skipping.")
            continue
        bench_df = pd.read_csv(path, parse_dates=["timestamp"])
        if bench_df["cpu_util"].max() > 1.5:
            bench_df["cpu_util"] = bench_df["cpu_util"] / 100.0
        bench_df["cpu_util"] = bench_df["cpu_util"].clip(0.0, 1.0).astype(np.float32)

        # Scale with model's fitted scaler
        raw_vals = bench_df["cpu_util"].values.reshape(-1, 1)
        scaled_bench = scaler.transform(raw_vals).ravel().astype(np.float32)
        X_bench, y_bench = make_windows(scaled_bench, window=W, horizon=H)

        b_res = evaluate_windows(
            model, scaler, X_bench, y_bench, cfg, device,
            label=label,
            output_plot=output_dir / plot_file,
            max_samples=max_samples or 1000,
            verbose=verbose,
        )
        all_results[label] = b_res

    if len(benchmarks_to_run) > 0 and verbose:
        print("\n" + "=" * 65)
        print("  Summary Benchmark Comparison")
        print("=" * 65)
        print(f"  {'Dataset':<28} {'MAE':>10} {'RMSE':>10} {'Coverage':>10}")
        print(f"  {'-'*62}")
        print(f"  {'Held-out Test Split':<28} {primary_res['mae']:>10.4f} {primary_res['rmse']:>10.4f} {primary_res['coverage']*100:>9.1f}%")
        for label, path, _ in benchmarks_to_run:
            if label in all_results:
                r = all_results[label]
                print(f"  {label:<28} {r['mae']:>10.4f} {r['rmse']:>10.4f} {r['coverage']*100:>9.1f}%")
        print("=" * 65)

    return primary_res


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the LSTM forecaster.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--benchmark", choices=["none", "azure", "alibaba", "all"], default="none",
        help="Evaluate on real cloud benchmark traces (azure, alibaba, all).",
    )
    parser.add_argument(
        "--eval-trace", default=None,
        help="Path to custom CSV trace for evaluation.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum windows to evaluate (useful for faster benchmark inference).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    cfg_path     = project_root / args.config

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    evaluate(
        cfg,
        benchmark=args.benchmark,
        eval_trace=args.eval_trace,
        max_samples=args.max_samples,
        verbose=True,
    )
