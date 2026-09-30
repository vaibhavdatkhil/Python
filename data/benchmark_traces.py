"""
data/benchmark_traces.py
────────────────────────
Real-world cloud workload trace generators and loaders modeled after:
  1. Azure Functions / VM Workload Dataset
  2. Alibaba Cluster Trace 2018/2021

Produces benchmark CSV traces adhering to the project schema:
  - timestamp : datetime (1-minute intervals)
  - cpu_util  : float in [0.0, 1.0]

Usage
─────
  python -m data.benchmark_traces --dataset azure --days 14
  python -m data.benchmark_traces --dataset alibaba --days 14
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_azure_trace(
    n_days: int = 14,
    freq: str = "1min",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic cloud workload trace modeled on Azure Functions / VM traces.

    Characteristics:
    - Multi-scale seasonality: 24-hour diurnal cycle + 7-day weekly cycle
    - Business-hour peaks (09:00 - 18:00 UTC) with lunch dip
    - Spiky micro-bursts (Poisson arrival process for flash crowds)
    - Autoregressive short-term memory (AR(1) process with high autocorrelation phi=0.92)
    - Weekend load reductions (~40% decrease)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(pd.Timedelta(days=n_days) / pd.Timedelta(freq))
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=n_steps, freq=freq)

    t = np.arange(n_steps)
    steps_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(freq))
    steps_per_week = steps_per_day * 7

    # 1. Base diurnal cycle (peak mid-afternoon, trough at 4 AM)
    hour_frac = (t % steps_per_day) / steps_per_day
    diurnal = 0.35 + 0.22 * np.sin(2 * np.pi * (hour_frac - 0.25))

    # 2. Secondary business-hours boost (bimodal: morning + afternoon peaks)
    biz_hours = np.exp(-0.5 * ((hour_frac - 0.45) / 0.12) ** 2) * 0.15 + \
                np.exp(-0.5 * ((hour_frac - 0.65) / 0.12) ** 2) * 0.18

    # 3. Weekly seasonality (weekdays higher, weekends lower)
    day_of_week = (t // steps_per_day) % 7
    weekend_mask = (day_of_week >= 5).astype(float)
    weekly_factor = 1.0 - 0.35 * weekend_mask

    # 4. Correlated AR(1) short-term fluctuations
    phi = 0.92
    ar_noise = np.zeros(n_steps, dtype=np.float32)
    innovations = rng.normal(0, 0.04, size=n_steps)
    for i in range(1, n_steps):
        ar_noise[i] = phi * ar_noise[i - 1] + innovations[i]

    # 5. Stochastic Flash Bursts (Micro-crowds / Batch jobs)
    # Poisson probability: ~8 burst events per day
    burst_prob = 8.0 / steps_per_day
    burst_events = rng.binomial(1, burst_prob, size=n_steps)
    burst_intensities = rng.exponential(scale=0.25, size=n_steps) * burst_events
    # Exponential decay over 15-30 steps
    burst_profile = np.zeros(n_steps, dtype=np.float32)
    decay = 0.82
    running_burst = 0.0
    for i in range(n_steps):
        running_burst = running_burst * decay + burst_intensities[i]
        burst_profile[i] = running_burst

    # Combine components
    cpu = (diurnal + biz_hours) * weekly_factor + ar_noise + burst_profile

    # Clip to valid utilization range [0.05, 0.98]
    cpu = np.clip(cpu, 0.05, 0.98).astype(np.float32)

    df = pd.DataFrame({"timestamp": timestamps, "cpu_util": cpu})
    return df


def generate_alibaba_trace(
    n_days: int = 14,
    freq: str = "1min",
    seed: int = 101,
) -> pd.DataFrame:
    """Generate a realistic cloud workload trace modeled on Alibaba Cluster 2018/2021 traces.

    Characteristics:
    - High-frequency batch job scheduling intervals (spikes every hour on the hour)
    - Heavy-tailed load spikes (e-commerce flash sales / promotional bursts)
    - Lower baseline utilization with sudden steep saturation periods
    """
    rng = np.random.default_rng(seed)
    n_steps = int(pd.Timedelta(days=n_days) / pd.Timedelta(freq))
    timestamps = pd.date_range("2026-02-01 00:00:00", periods=n_steps, freq=freq)

    t = np.arange(n_steps)
    steps_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(freq))
    steps_per_hour = int(pd.Timedelta(hours=1) / pd.Timedelta(freq))

    # Base utilization
    base = 0.28 + 0.15 * np.sin(2 * np.pi * (t % steps_per_day) / steps_per_day)

    # Hourly batch job cron spikes
    minute_in_hour = t % steps_per_hour
    hourly_cron = np.exp(-0.5 * (minute_in_hour / 2.0) ** 2) * 0.22

    # Heavy-tailed Pareto bursts (flash sale traffic)
    pareto_prob = 4.0 / steps_per_day
    pareto_mask = rng.binomial(1, pareto_prob, size=n_steps)
    pareto_spikes = (rng.pareto(a=3.0, size=n_steps) * 0.15) * pareto_mask

    # Correlated background noise
    phi = 0.88
    noise = np.zeros(n_steps, dtype=np.float32)
    innov = rng.normal(0, 0.035, size=n_steps)
    for i in range(1, n_steps):
        noise[i] = phi * noise[i - 1] + innov[i]

    cpu = base + hourly_cron + pareto_spikes + noise
    cpu = np.clip(cpu, 0.04, 0.99).astype(np.float32)

    return pd.DataFrame({"timestamp": timestamps, "cpu_util": cpu})


def save_benchmark_trace(df: pd.DataFrame, out_path: Path | str) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark cloud traces")
    parser.add_argument("--dataset", choices=["azure", "alibaba", "both"], default="both")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--freq", type=str, default="1min")
    args = parser.parse_args()

    data_dir = Path(__file__).parent

    if args.dataset in ["azure", "both"]:
        df_azure = generate_azure_trace(n_days=args.days, freq=args.freq)
        p_azure = save_benchmark_trace(df_azure, data_dir / "azure_vm_workload_trace.csv")
        print(f"[OK] Generated Azure benchmark trace: {p_azure} ({len(df_azure)} rows, mean CPU={df_azure['cpu_util'].mean():.3f})")

    if args.dataset in ["alibaba", "both"]:
        df_ali = generate_alibaba_trace(n_days=args.days, freq=args.freq)
        p_ali = save_benchmark_trace(df_ali, data_dir / "alibaba_cluster_trace.csv")
        print(f"[OK] Generated Alibaba benchmark trace: {p_ali} ({len(df_ali)} rows, mean CPU={df_ali['cpu_util'].mean():.3f})")
