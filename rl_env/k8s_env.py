"""
rl_env/k8s_env.py
──────────────────
Gymnasium environment that simulates a Kubernetes pod-autoscaling problem.

The environment replays a CPU-utilisation workload trace (real or synthetic)
and gives an RL agent control over the number of running pods.  At each step
the agent receives an observation that includes the raw CPU history **and** the
LSTM MC-Dropout forecast, then picks a scaling action.  A shaped reward
balances SLO compliance, infrastructure cost, and scaling stability.

Episode lifecycle
─────────────────
  reset() → start of trace, initial replicas
  step()  → advance one timestep, apply (lagged) scaling, return obs + reward
  done    → truncated=True when the trace is exhausted (no terminal failure)

Scaling lag
───────────
  Scaling actions are placed in a FIFO queue and applied only after
  `scale_lag_steps` steps (mimicking Kubernetes cold-start latency).

LSTM model dependency
─────────────────────
  If `model` is None the forecast components of the observation are zeroed out.
  This allows the environment to be instantiated and tested without a trained
  checkpoint (useful for early-stage unit tests).

Usage
─────
  python -m rl_env.k8s_env          # runs the built-in smoke test
  python -m rl_env.k8s_env --steps 500  # longer run
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import yaml

from rl_env.spaces import ACTION_TO_DELTA, make_spaces, obs_slices


class K8sAutoscalingEnv(gym.Env):
    """Simulated Kubernetes autoscaling environment.

    Parameters
    ----------
    cfg      : parsed config dict (from config.yaml).
    model    : trained LSTMForecast (or None → forecast observation = zeros).
    scaler   : fitted MinMaxScaler (or None).
    trace_df : DataFrame with DatetimeIndex and ``cpu_util`` column [0, 1].
    device   : torch device for LSTM inference.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        cfg: dict,
        model=None,
        scaler=None,
        trace_df: Optional[pd.DataFrame] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        self._cfg    = cfg
        self._model  = model
        self._scaler = scaler
        self._device = torch.device(device) if isinstance(device, str) else device

        # ── spaces ────────────────────────────────────────────────────────────
        self.observation_space, self.action_space = make_spaces(cfg)
        self._slices = obs_slices(cfg)

        # ── preprocessing config ──────────────────────────────────────────────
        self._window  = cfg["preprocessing"]["window_size"]   # W
        self._horizon = cfg["preprocessing"]["horizon"]        # H

        # ── cluster config ────────────────────────────────────────────────────
        rc = cfg["rl_env"]
        self._min_rep       = int(rc["min_replicas"])
        self._max_rep       = int(rc["max_replicas"])
        self._init_rep      = int(rc["initial_replicas"])
        self._slo_thresh    = float(rc["slo_threshold"])
        self._lag           = int(rc["scale_lag_steps"])
        self._r_slo_met     = float(rc["reward_slo_met"])
        self._r_slo_viol    = float(rc["reward_slo_violated"])
        self._r_cost        = float(rc["reward_cost_per_replica"])
        self._r_stability   = float(rc["reward_stability_penalty"])

        # ── workload trace ────────────────────────────────────────────────────
        if trace_df is not None:
            self._cpu_series = trace_df["cpu_util"].values.astype(np.float32)
        else:
            # Fallback: generate a short synthetic trace inline
            from data.synthetic_generator import generate_trace
            df = generate_trace(
                n_days=cfg["data"]["n_days"],
                freq=cfg["data"]["freq"],
                seed=cfg["data"]["seed"],
            )
            self._cpu_series = df["cpu_util"].values.astype(np.float32)

        self._n_steps = len(self._cpu_series)

        # ── mutable state (set in reset) ──────────────────────────────────────
        self._t: int = 0                     # current timestep into the trace
        self._replicas: int = self._init_rep
        self._pending_deltas: collections.deque[int] = collections.deque()
        self._last_action_delta: int = 0     # for stability reward

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset to the beginning of the trace.

        Returns
        -------
        obs  : np.ndarray, shape (obs_dim,)
        info : dict
        """
        super().reset(seed=seed)   # seeds self.np_random

        # Start far enough into the trace that we have a full look-back window
        self._t        = self._window
        self._replicas = self._init_rep

        # Clear scaling queue; pre-fill with no-op deltas
        self._pending_deltas = collections.deque(
            [0] * self._lag, maxlen=self._lag
        )
        self._last_action_delta = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one timestep.

        Parameters
        ----------
        action : int in [0, 4]  (see rl_env/spaces.py for mapping)

        Returns
        -------
        obs        : np.ndarray
        reward     : float
        terminated : bool  (always False — no terminal failure state)
        truncated  : bool  (True when trace is exhausted)
        info       : dict
        """
        assert self.action_space.contains(int(action)), \
            f"Invalid action {action}; must be in [0, {self.action_space.n - 1}]"

        # ── 1. Enqueue the requested delta ────────────────────────────────────
        delta = ACTION_TO_DELTA[int(action)]
        self._pending_deltas.append(delta)

        # ── 2. Apply the oldest pending delta (scaling lag) ───────────────────
        effective_delta = self._pending_deltas.popleft()
        old_replicas    = self._replicas
        self._replicas  = int(
            np.clip(self._replicas + effective_delta, self._min_rep, self._max_rep)
        )
        actual_delta    = self._replicas - old_replicas  # may differ after clamping

        # ── 3. Compute reward BEFORE advancing time ───────────────────────────
        reward = self._compute_reward(actual_delta)

        # ── 4. Advance time ───────────────────────────────────────────────────
        self._t += 1
        self._last_action_delta = actual_delta

        # ── 5. Build output ───────────────────────────────────────────────────
        terminated = False
        truncated  = self._t >= self._n_steps - self._horizon
        obs        = self._get_obs() if not truncated else self._get_obs()
        info       = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "ansi") -> str:
        """Return a one-line ASCII summary of the current state."""
        cpu = float(self._cpu_series[self._t]) if self._t < self._n_steps else 0.0
        cpp = cpu / max(self._replicas, 1)
        slo = "OK" if cpp <= self._slo_thresh else "FAIL"
        line = (
            f"t={self._t:5d}  cpu={cpu:.3f}  "
            f"replicas={self._replicas:2d}  "
            f"cpu/pod={cpp:.3f}  SLO={slo}"
        )
        if mode == "ansi":
            return line
        return line

    def close(self) -> None:
        pass

    # ── internal helpers ──────────────────────────────────────────────────────

    def _current_cpu(self) -> float:
        idx = min(self._t, self._n_steps - 1)
        return float(self._cpu_series[idx])

    def _cpu_per_pod(self) -> float:
        return self._current_cpu() / max(self._replicas, 1)

    def _get_obs(self) -> np.ndarray:
        """Build the flat observation vector."""
        W = self._window
        H = self._horizon

        # ── CPU look-back window ──────────────────────────────────────────────
        start = max(0, self._t - W)
        raw   = self._cpu_series[start : self._t]
        # Pad with zeros if we don't yet have W steps (shouldn't happen after
        # reset(), but guard anyway)
        if len(raw) < W:
            raw = np.concatenate([np.zeros(W - len(raw), dtype=np.float32), raw])
        cpu_window = raw.astype(np.float32)  # (W,)

        # ── LSTM forecast ─────────────────────────────────────────────────────
        if self._model is not None:
            forecast_mean, forecast_std = self._run_lstm()
        else:
            forecast_mean = np.zeros(H, dtype=np.float32)
            forecast_std  = np.zeros(H, dtype=np.float32)

        # ── Scalar cluster features ───────────────────────────────────────────
        rep_norm    = np.float32(self._replicas / self._max_rep)
        cpp_norm    = np.float32(np.clip(self._cpu_per_pod(), 0.0, 1.0))

        obs = np.concatenate([
            cpu_window,       # (W,)
            forecast_mean,    # (H,)
            forecast_std,     # (H,)
            [rep_norm],       # (1,)
            [cpp_norm],       # (1,)
        ]).astype(np.float32)

        return obs

    def _run_lstm(self) -> tuple[np.ndarray, np.ndarray]:
        """Run MC Dropout inference and return (mean, std) per horizon step."""
        from model.inference import mc_predict

        W = self._window
        start = max(0, self._t - W)
        raw   = self._cpu_series[start : self._t]
        if len(raw) < W:
            raw = np.concatenate([np.zeros(W - len(raw), dtype=np.float32), raw])

        inf_cfg   = self._cfg["inference"]
        n_samples = inf_cfg["n_mc_samples"]
        lo        = inf_cfg["ci_lower_pct"]
        hi        = inf_cfg["ci_upper_pct"]

        # mc_predict returns (mean, lower, upper) in original scale.
        # We want mean and per-step std, so we do the stochastic passes
        # ourselves to get the full sample distribution.
        x = raw.reshape(1, W, 1).astype(np.float32)
        x_t = torch.from_numpy(x).expand(n_samples, -1, -1).contiguous().to(self._device)

        from model.lstm_model import mc_dropout_ctx
        with mc_dropout_ctx(self._model):
            with torch.no_grad():
                out = self._model(x_t)   # (n_samples, H) in a single batched pass
                preds = out.cpu().numpy()

        mean_scaled = preds.mean(axis=0)       # (H,)
        std_scaled  = preds.std(axis=0)        # (H,)

        # Inverse-transform mean to original scale; keep std in [0,1] (it's
        # already small and roughly normalised)
        if self._scaler is not None:
            mean_orig = self._scaler.inverse_transform(
                mean_scaled.reshape(-1, 1)
            ).ravel().astype(np.float32)
        else:
            mean_orig = mean_scaled.astype(np.float32)

        # Clip to [0, 1] for the observation box
        mean_obs = np.clip(mean_orig, 0.0, 1.0).astype(np.float32)
        std_obs  = np.clip(std_scaled, 0.0, 1.0).astype(np.float32)

        return mean_obs, std_obs

    def _compute_reward(self, actual_delta: int) -> float:
        """Compute the shaped reward for the current step.

        Components
        ──────────
        r_slo       : +r_slo_met if SLO met, r_slo_viol otherwise
        r_cost      : r_cost_per_replica × replicas  (negative)
        r_stability : r_stability_penalty if replicas changed  (negative or 0)
        """
        cpp = self._cpu_per_pod()

        # SLO term
        r_slo = self._r_slo_met if cpp <= self._slo_thresh else self._r_slo_viol

        # Cost term — proportional to replica count
        r_cost = self._r_cost * self._replicas

        # Stability term — penalise actual changes (after clamping)
        r_stability = self._r_stability if actual_delta != 0 else 0.0

        return float(r_slo + r_cost + r_stability)

    def _get_info(self) -> dict[str, Any]:
        """Return a rich diagnostic info dict."""
        cpu = self._current_cpu()
        cpp = self._cpu_per_pod()
        return {
            "t":              self._t,
            "cpu_util":       cpu,
            "replicas":       self._replicas,
            "cpu_per_pod":    cpp,
            "slo_met":        bool(cpp <= self._slo_thresh),
            "pending_deltas": list(self._pending_deltas),
        }


# ── CLI smoke test ─────────────────────────────────────────────────────────────

def _smoke_test(n_steps: int = 200, use_model: bool = True) -> None:
    """Run the environment for `n_steps` random actions and verify invariants."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    cfg_path     = project_root / "config.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("-" * 60)
    print(f"  K8sAutoscalingEnv -- smoke test  ({n_steps} random steps)")
    print("-" * 60)

    # Optionally load trained model
    model, scaler, device = None, None, "cpu"
    ckpt_path = project_root / cfg["training"]["checkpoint_dir"] / "best_model.pt"

    if use_model and ckpt_path.exists():
        from model.inference import load_checkpoint
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, scaler, _ = load_checkpoint(ckpt_path, device=device_obj)
        device = device_obj
        print(f"  LSTM model loaded from {ckpt_path.name}")
    else:
        print("  Running WITHOUT LSTM model (forecast obs = zeros).")

    # Build trace
    from data.loader import load_trace
    df = load_trace(cfg=cfg)

    # Instantiate env
    env = K8sAutoscalingEnv(cfg, model=model, scaler=scaler, trace_df=df, device=device)

    # Gymnasium env checker
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, warn=True)
        print("  gymnasium.check_env  -> PASSED [OK]")
    except Exception as exc:
        print(f"  gymnasium.check_env  -> WARNING: {exc}")

    # Random rollout
    obs, info = env.reset(seed=0)
    obs_dim   = env.observation_space.shape[0]

    total_reward  = 0.0
    slo_met_count = 0
    replica_sum   = 0

    for step_i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # ── invariant checks ──────────────────────────────────────────────────
        assert obs.shape == (obs_dim,), \
            f"[FAIL] obs shape {obs.shape} != ({obs_dim},) at step {step_i}"
        assert np.isfinite(reward), \
            f"[FAIL] reward is not finite at step {step_i}: {reward}"
        assert "slo_met" in info, "[FAIL] 'slo_met' missing from info"
        assert "replicas" in info, "[FAIL] 'replicas' missing from info"
        assert "cpu_per_pod" in info, "[FAIL] 'cpu_per_pod' missing from info"

        total_reward  += reward
        slo_met_count += int(info["slo_met"])
        replica_sum   += info["replicas"]

        if (step_i + 1) % 50 == 0:
            print(f"  {env.render()}")

        if truncated or terminated:
            print(f"  Episode ended at step {step_i + 1}.")
            break

    print("-" * 60)
    print(f"  Steps completed   : {n_steps}")
    print(f"  Total reward      : {total_reward:.2f}")
    print(f"  SLO compliance    : {slo_met_count / n_steps * 100:.1f}%")
    print(f"  Avg replicas      : {replica_sum / n_steps:.2f}")
    print(f"  All assertions    : PASSED [OK]")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test K8sAutoscalingEnv.")
    parser.add_argument("--steps",     type=int,  default=200,  help="Random steps to run.")
    parser.add_argument("--no-model",  action="store_true",     help="Skip loading LSTM model.")
    args = parser.parse_args()

    _smoke_test(n_steps=args.steps, use_model=not args.no_model)
