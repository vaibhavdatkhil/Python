"""
rl_env/spaces.py
─────────────────
Constructs the Gymnasium observation_space and action_space for
K8sAutoscalingEnv, derived entirely from config.yaml values so the
spaces stay consistent with the rest of the pipeline.

Observation layout (all floats, Box):
  ┌──────────────────────────────┬──────────┬──────────────────────────────┐
  │ Slice                        │ Size     │ Description                  │
  ├──────────────────────────────┼──────────┼──────────────────────────────┤
  │ obs[0 : W]                   │ W=60     │ Normalised CPU window [0,1]  │
  │ obs[W : W+H]                 │ H=15     │ LSTM forecast mean  [0,1]    │
  │ obs[W+H : W+2H]              │ H=15     │ LSTM forecast std   [0,1]    │
  │ obs[W+2H]                    │ 1        │ current_replicas / max_rep.  │
  │ obs[W+2H+1]                  │ 1        │ cpu_per_pod (clipped [0,1])  │
  └──────────────────────────────┴──────────┴──────────────────────────────┘
  Total: W + 2H + 2 = 92 for defaults (W=60, H=15).

Action layout (Discrete):
  0 → scale down 2   (Δ = −2)
  1 → scale down 1   (Δ = −1)
  2 → no-op          (Δ =  0)
  3 → scale up 1     (Δ = +1)
  4 → scale up 2     (Δ = +2)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Maps discrete action index → replica delta
ACTION_TO_DELTA: dict[int, int] = {
    0: -2,
    1: -1,
    2:  0,
    3: +1,
    4: +2,
}

N_ACTIONS = len(ACTION_TO_DELTA)


def make_spaces(cfg: dict) -> tuple[spaces.Box, spaces.Discrete]:
    """Build observation_space and action_space from config.

    Parameters
    ----------
    cfg : parsed config dict (from config.yaml).

    Returns
    -------
    observation_space : gym.spaces.Box, shape (obs_dim,), dtype float32
    action_space      : gym.spaces.Discrete(5)
    """
    window  = cfg["preprocessing"]["window_size"]   # W
    horizon = cfg["preprocessing"]["horizon"]        # H

    obs_dim = window + 2 * horizon + 2

    observation_space = spaces.Box(
        low=np.zeros(obs_dim, dtype=np.float32),
        high=np.ones(obs_dim, dtype=np.float32),
        dtype=np.float32,
    )
    action_space = spaces.Discrete(N_ACTIONS)
    return observation_space, action_space


def obs_slices(cfg: dict) -> dict[str, slice | int]:
    """Return named slices into the flat observation vector.

    Useful for inspecting / logging individual components.
    """
    W = cfg["preprocessing"]["window_size"]
    H = cfg["preprocessing"]["horizon"]
    return {
        "cpu_window":       slice(0, W),
        "forecast_mean":    slice(W, W + H),
        "forecast_std":     slice(W + H, W + 2 * H),
        "replicas_norm":    W + 2 * H,
        "cpu_per_pod_norm": W + 2 * H + 1,
    }
