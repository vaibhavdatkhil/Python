"""
data/synthetic_generator.py
───────────────────────────
Generates a synthetic cloud workload trace that mimics the statistical
properties of real Azure Functions / cloud CPU-utilisation traces:

  • Daily sinusoidal pattern   (peak at business hours)
  • Weekly sinusoidal pattern  (lower traffic on weekends)
  • Poisson-distributed random bursts (amplitude + duration randomised)
  • Additive white Gaussian noise

Output schema
─────────────
  timestamp : DatetimeTZNaive index (or column), freq=`freq`
  cpu_util  : float in [0, 1]  — normalised CPU utilisation

This schema is identical to the one expected from a real CSV trace, so the
data/loader.py interface is a transparent drop-in replacement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_trace(
    n_days: int = 30,
    freq: str = "1min",
    seed: int = 42,
    burst_rate: float = 3.0,      # expected bursts per day (Poisson λ)
    burst_amp_range: tuple[float, float] = (0.2, 0.5),
    burst_dur_range: tuple[int, int] = (5, 60),   # in timesteps
    noise_std: float = 0.03,
) -> pd.DataFrame:
    """Return a DataFrame with columns [timestamp, cpu_util].

    Parameters
    ----------
    n_days:
        Length of the trace in days.
    freq:
        Pandas frequency alias for the desired time resolution (e.g. "1min").
    seed:
        Random seed for reproducibility.
    burst_rate:
        Expected number of traffic bursts per day (Poisson process).
    burst_amp_range:
        (min, max) amplitude of each burst as a fraction of the [0,1] scale.
    burst_dur_range:
        (min, max) duration of each burst in timesteps.
    noise_std:
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (datetime64), cpu_util (float32, clipped [0,1]).
    """
    rng = np.random.default_rng(seed)

    # ── 1. Time axis ──────────────────────────────────────────────────────────
    start = pd.Timestamp("2024-01-01")
    index = pd.date_range(start=start, periods=n_days * 24 * 60, freq=freq)
    n = len(index)
    t = np.arange(n, dtype=np.float64)

    # ── 2. Seasonal components ────────────────────────────────────────────────
    # Seconds per timestep (for freq="1min" -> 60 s)
    step_seconds = pd.tseries.frequencies.to_offset(freq).nanos / 1e9

    daily_period  = 24 * 3600 / step_seconds   # timesteps in one day
    weekly_period = 7 * daily_period            # timesteps in one week

    # Daily: peaks around t=9h (business-hours ramp-up)
    daily_phase = 2 * np.pi * 9 / 24           # phase shift -> peak at 9 AM
    daily  = 0.35 * np.sin(2 * np.pi * t / daily_period  - daily_phase) + 0.35

    # Weekly: lower over weekends (amplitude modulation)
    weekly = 0.10 * np.sin(2 * np.pi * t / weekly_period - np.pi / 2)

    signal = daily + weekly  # base signal in roughly [0, 0.8]

    # ── 3. Burst injection (Poisson arrival, random amp & duration) ───────────
    total_bursts = int(rng.poisson(burst_rate * n_days))
    burst_starts = rng.integers(0, n, size=total_bursts)

    for start_idx in burst_starts:
        amp = rng.uniform(*burst_amp_range)
        dur = int(rng.integers(*burst_dur_range))
        end_idx = min(start_idx + dur, n)
        # Smooth burst shape: raised cosine envelope
        burst_len = end_idx - start_idx
        envelope = amp * 0.5 * (1 - np.cos(np.pi * np.arange(burst_len) / burst_len))
        signal[start_idx:end_idx] += envelope

    # ── 4. Noise ─────────────────────────────────────────────────────────────
    signal += rng.normal(0.0, noise_std, size=n)

    # ── 5. Clip to [0, 1] and build DataFrame ─────────────────────────────────
    cpu_util = np.clip(signal, 0.0, 1.0).astype(np.float32)

    df = pd.DataFrame({"timestamp": index, "cpu_util": cpu_util})
    return df


# ── CLI sanity check ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_trace(n_days=7)
    print(f"Generated trace: {df.shape}")
    print(df.head(10).to_string(index=False))
    print(f"\ncpu_util stats:\n{df['cpu_util'].describe()}")
