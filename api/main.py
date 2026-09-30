"""
api/main.py
────────────
Phase 9: FastAPI REST backend for the K8s RL Autoscaling system.

Endpoints
─────────
  GET  /health          — liveness probe
  GET  /config          — expose relevant config values
  POST /forecast        — LSTM MC-Dropout point forecast + CI band
  POST /scale-action    — PPO agent picks a scaling action
  GET  /metrics         — latest episode metrics summary
  POST /run-episode     — run one full PPO episode and update metrics

Running
───────
  uvicorn api.main:app --reload --port 8000
  # Or from project root:
  python -m api.main

API docs (auto-generated)
─────────────────────────
  http://localhost:8000/docs    (Swagger UI)
  http://localhost:8000/redoc   (ReDoc)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ForecastRequest, ForecastResponse,
    ScaleRequest, ScaleResponse,
    MetricsResponse, ConfigResponse,
)

# ── startup state (module-level singletons) ───────────────────────────────────

_cfg:         Optional[dict]  = None
_device:      Optional[torch.device] = None
_lstm_model                    = None   # LSTMForecast | None
_scaler                        = None   # MinMaxScaler | None
_ppo_policy                    = None   # ActorCritic  | None
_trace_df                      = None   # pd.DataFrame | None

# Metrics state (updated by /run-episode)
_metrics_state = {
    "total_reward":  None,
    "slo_pct":       None,
    "avg_replicas":  None,
    "episode_count": 0,
}

# Action index → delta replicas
_ACTION_TO_DELTA = {0: -2, 1: -1, 2: 0, 3: +1, 4: +2}
_ACTION_LABELS   = {0: "Scale -2", 1: "Scale -1", 2: "Hold", 3: "Scale +1", 4: "Scale +2"}


# ── lifespan: load models at startup ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config, LSTM model, and PPO agent at server startup."""
    global _cfg, _device, _lstm_model, _scaler, _ppo_policy, _trace_df

    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        _cfg = yaml.safe_load(f)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[API] Device: {_device}", flush=True)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    lstm_ckpt = PROJECT_ROOT / _cfg["training"]["checkpoint_dir"] / "best_model.pt"
    if lstm_ckpt.exists():
        try:
            from model.inference import load_checkpoint as _load_lstm
            _lstm_model, _scaler, _ = _load_lstm(lstm_ckpt, device=_device)
            print(f"[API] LSTM model loaded from {lstm_ckpt.name}", flush=True)
        except Exception as exc:
            print(f"[API][WARN] Could not load LSTM: {exc}", flush=True)
    else:
        print("[API][WARN] LSTM checkpoint not found — /forecast will return zeros.", flush=True)

    # ── PPO ───────────────────────────────────────────────────────────────────
    ckpt_dir = PROJECT_ROOT / _cfg["ppo"]["checkpoint_dir"]
    ppo_ckpt = ckpt_dir / "ppo_agent.pt"
    if not ppo_ckpt.exists():
        ppo_ckpt = ckpt_dir / "ppo_agent_final.pt"
    if ppo_ckpt.exists():
        try:
            from rl_agent.ppo import load_ppo_checkpoint
            _ppo_policy, _ = load_ppo_checkpoint(ppo_ckpt, device=_device)
            print(f"[API] PPO policy loaded from {ppo_ckpt.name}", flush=True)
        except Exception as exc:
            print(f"[API][WARN] Could not load PPO: {exc}", flush=True)
    else:
        print("[API][WARN] PPO checkpoint not found — /scale-action will return hold(0).", flush=True)

    # ── trace ─────────────────────────────────────────────────────────────────
    try:
        from data.loader import load_trace
        _trace_df = load_trace(cfg=_cfg)
        print(f"[API] Trace loaded: {len(_trace_df):,} timesteps", flush=True)
    except Exception as exc:
        print(f"[API][WARN] Could not load trace: {exc}", flush=True)

    yield  # server running

    print("[API] Shutdown.", flush=True)


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="K8s RL Autoscaling API",
    description=(
        "REST backend for the K8s RL Autoscaling B.Tech project.\n\n"
        "Exposes the LSTM + MC-Dropout forecaster and the PPO autoscaling agent "
        "as REST endpoints for integration with dashboards, Kubernetes controllers, "
        "or any external system."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Liveness probe",
    tags=["System"],
)
def health():
    """Return server status and which models are loaded."""
    return {
        "status":       "ok",
        "model_loaded": _lstm_model is not None,
        "agent_loaded": _ppo_policy is not None,
        "device":       str(_device),
        "timestamp":    time.time(),
    }


# ── GET /config ───────────────────────────────────────────────────────────────

@app.get(
    "/config",
    response_model=ConfigResponse,
    summary="Expose relevant config values",
    tags=["System"],
)
def get_config():
    """Return the configuration values relevant to the dashboard."""
    if _cfg is None:
        raise HTTPException(status_code=503, detail="Config not loaded.")
    return ConfigResponse(
        window_size   = _cfg["preprocessing"]["window_size"],
        horizon       = _cfg["preprocessing"]["horizon"],
        min_replicas  = _cfg["rl_env"]["min_replicas"],
        max_replicas  = _cfg["rl_env"]["max_replicas"],
        slo_threshold = _cfg["rl_env"]["slo_threshold"],
        n_mc_samples  = _cfg["inference"]["n_mc_samples"],
        ci_lower_pct  = _cfg["inference"]["ci_lower_pct"],
        ci_upper_pct  = _cfg["inference"]["ci_upper_pct"],
    )


# ── POST /forecast ────────────────────────────────────────────────────────────

@app.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="LSTM MC-Dropout forecast",
    tags=["Forecasting"],
)
def forecast(req: ForecastRequest):
    """Run the LSTM + MC-Dropout forecaster on a given CPU window.

    Returns mean forecast plus 5th–95th percentile confidence interval
    for the next `horizon` timesteps (original CPU-utilisation scale).
    """
    if _cfg is None:
        raise HTTPException(status_code=503, detail="Server not ready.")

    W         = _cfg["preprocessing"]["window_size"]
    n_samples = req.n_samples or _cfg["inference"]["n_mc_samples"]
    lo_pct    = req.ci_lower  or _cfg["inference"]["ci_lower_pct"]
    hi_pct    = req.ci_upper  or _cfg["inference"]["ci_upper_pct"]

    if len(req.cpu_window) != W:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"cpu_window must have exactly {W} values; got {len(req.cpu_window)}.",
        )

    cpu_arr = np.array(req.cpu_window, dtype=np.float32)

    if _lstm_model is None:
        # Return zeros if no model
        H = _cfg["preprocessing"]["horizon"]
        return ForecastResponse(
            horizon=H, mean=[0.0]*H, lower=[0.0]*H, upper=[0.0]*H,
            n_samples_used=0, model_loaded=False,
        )

    try:
        from model.inference import mc_predict
        mean, lower, upper = mc_predict(
            _lstm_model, cpu_arr,
            n_samples=n_samples,
            scaler=_scaler,
            lower_pct=lo_pct,
            upper_pct=hi_pct,
            device=_device,
        )
        return ForecastResponse(
            horizon=len(mean),
            mean=mean.tolist(),
            lower=lower.tolist(),
            upper=upper.tolist(),
            n_samples_used=n_samples,
            model_loaded=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


# ── POST /scale-action ────────────────────────────────────────────────────────

@app.post(
    "/scale-action",
    response_model=ScaleResponse,
    summary="PPO agent scaling decision",
    tags=["RL Agent"],
)
def scale_action(req: ScaleRequest):
    """Query the trained PPO agent for a scaling action.

    Constructs the observation vector from the provided fields and
    returns the discrete action chosen by the policy.
    """
    if _cfg is None:
        raise HTTPException(status_code=503, detail="Server not ready.")

    W = _cfg["preprocessing"]["window_size"]
    H = _cfg["preprocessing"]["horizon"]

    if len(req.cpu_window) != W:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"cpu_window must have exactly {W} values; got {len(req.cpu_window)}.",
        )

    # Build observation vector
    forecast_mean = np.array(req.forecast_mean or [0.0]*H, dtype=np.float32)
    forecast_std  = np.array(req.forecast_std  or [0.0]*H, dtype=np.float32)

    obs = np.concatenate([
        np.array(req.cpu_window, dtype=np.float32),
        forecast_mean[:H],
        forecast_std[:H],
        [req.replicas_norm],
        [req.cpu_per_pod_norm],
    ]).astype(np.float32)

    if _ppo_policy is None:
        # Fallback: return hold action
        return ScaleResponse(
            action=2, delta=0, action_label="Hold (no agent loaded)",
            agent_loaded=False,
        )

    try:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(_device)
        with torch.no_grad():
            action_t, _, _, _ = _ppo_policy.act(obs_t)
            action = int(action_t.item())
        delta = _ACTION_TO_DELTA[action]
        return ScaleResponse(
            action=action,
            delta=delta,
            action_label=_ACTION_LABELS[action],
            agent_loaded=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Policy error: {exc}")


# ── GET /metrics ──────────────────────────────────────────────────────────────

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Latest episode metrics",
    tags=["RL Agent"],
)
def get_metrics():
    """Return the metrics from the most recently completed episode.

    Call POST /run-episode first to populate these values.
    """
    return MetricsResponse(
        total_reward  = _metrics_state["total_reward"],
        slo_pct       = _metrics_state["slo_pct"],
        avg_replicas  = _metrics_state["avg_replicas"],
        episode_count = _metrics_state["episode_count"],
        agent_loaded  = _ppo_policy  is not None,
        model_loaded  = _lstm_model  is not None,
    )


# ── POST /run-episode ─────────────────────────────────────────────────────────

@app.post(
    "/run-episode",
    summary="Run one full PPO episode and update metrics",
    tags=["RL Agent"],
)
def run_episode(seed: int = 42):
    """Run a complete PPO episode in the Gymnasium environment.

    Updates the /metrics endpoint with total reward, SLO compliance %,
    and average replica count.  May take several seconds.
    """
    if _trace_df is None:
        raise HTTPException(status_code=503, detail="Trace not loaded.")
    if _cfg is None:
        raise HTTPException(status_code=503, detail="Config not loaded.")

    try:
        from rl_env.k8s_env import K8sAutoscalingEnv
        env = K8sAutoscalingEnv(
            cfg=_cfg, model=_lstm_model, scaler=_scaler,
            trace_df=_trace_df, device=_device,
        )
        obs_np, _ = env.reset(seed=seed)
        obs = torch.from_numpy(obs_np).float().to(_device)

        total_reward = 0.0
        slo_count    = 0
        replica_sum  = 0
        n_steps      = 0

        done = False
        while not done:
            if _ppo_policy is not None:
                with torch.no_grad():
                    action_t, _, _, _ = _ppo_policy.act(obs.unsqueeze(0))
                    action = int(action_t.item())
            else:
                action = env.action_space.sample()

            obs_np, reward, terminated, truncated, info = env.step(action)
            obs  = torch.from_numpy(obs_np).float().to(_device)
            done = terminated or truncated

            total_reward += float(reward)
            slo_count    += int(info.get("slo_met", False))
            replica_sum  += int(info.get("replicas", 1))
            n_steps      += 1

        env.close()

        slo_pct      = slo_count / max(n_steps, 1) * 100
        avg_replicas = replica_sum / max(n_steps, 1)

        _metrics_state["total_reward"]  = round(total_reward, 2)
        _metrics_state["slo_pct"]       = round(slo_pct, 2)
        _metrics_state["avg_replicas"]  = round(avg_replicas, 3)
        _metrics_state["episode_count"] += 1

        return {
            "total_reward": round(total_reward, 2),
            "slo_pct":      round(slo_pct, 2),
            "avg_replicas": round(avg_replicas, 3),
            "steps":        n_steps,
            "agent_used":   _ppo_policy is not None,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Episode error: {exc}")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    api_cfg = _cfg.get("api", {}) if _cfg else {}
    host    = api_cfg.get("host", "127.0.0.1")
    port    = api_cfg.get("port", 8000)
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
