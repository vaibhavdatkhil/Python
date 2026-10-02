"""
rl_env — Gymnasium RL environment for K8s autoscaling (Phase 6).

Public API
──────────
  K8sAutoscalingEnv  : the main Gymnasium environment.
  RecordEpisodeStats : wrapper that accumulates per-episode statistics.
  TransitionLogger   : wrapper that logs (s, a, r, s', done) to CSV.

Usage
─────
  from rl_env import K8sAutoscalingEnv
  env = K8sAutoscalingEnv(cfg, model=model, scaler=scaler, trace_df=df, device=device)
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(action)
"""

from rl_env.k8s_env import K8sAutoscalingEnv
from rl_env.wrappers import RecordEpisodeStats, TransitionLogger

__all__ = ["K8sAutoscalingEnv", "RecordEpisodeStats", "TransitionLogger"]
