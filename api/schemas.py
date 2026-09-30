"""
api/schemas.py
───────────────
Pydantic request and response models for the FastAPI backend.

All models use strict typing and include docstrings / examples
for automatic OpenAPI documentation.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── /forecast ─────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """Request body for POST /forecast.

    Attributes
    ----------
    cpu_window : list of CPU-utilisation values (0–1), length = window_size.
    n_samples  : MC-Dropout samples to draw (default: config value).
    ci_lower   : lower CI percentile (default 5).
    ci_upper   : upper CI percentile (default 95).
    """
    cpu_window: List[float] = Field(
        ...,
        description="CPU-utilisation look-back window (0–1 normalised), "
                    "length must equal preprocessing.window_size in config.",
        examples=[[0.3, 0.35, 0.4, 0.45]],
    )
    n_samples: Optional[int] = Field(
        default=None,
        description="MC-Dropout forward passes. Defaults to config value.",
        ge=1, le=500,
    )
    ci_lower: Optional[int] = Field(
        default=None,
        description="Lower CI percentile (e.g. 5 → 5th percentile).",
        ge=0, le=49,
    )
    ci_upper: Optional[int] = Field(
        default=None,
        description="Upper CI percentile (e.g. 95 → 95th percentile).",
        ge=51, le=100,
    )


class ForecastResponse(BaseModel):
    """Response body for POST /forecast."""
    horizon: int = Field(..., description="Number of forecast steps.")
    mean:    List[float] = Field(..., description="Mean forecast (original scale).")
    lower:   List[float] = Field(..., description="Lower CI bound (original scale).")
    upper:   List[float] = Field(..., description="Upper CI bound (original scale).")
    n_samples_used: int  = Field(..., description="MC-Dropout samples used.")
    model_loaded:   bool = Field(..., description="Whether the LSTM model was loaded.")


# ── /scale-action ─────────────────────────────────────────────────────────────

class ScaleRequest(BaseModel):
    """Request body for POST /scale-action.

    The observation vector must have the same dimension as what the PPO agent
    was trained on: [cpu_window(W), forecast_mean(H), forecast_std(H),
    replicas_norm(1), cpu_per_pod_norm(1)].

    Alternatively, provide the raw fields and the server will build the vector.
    """
    cpu_window:       List[float] = Field(
        ...,
        description="CPU look-back window (length W, 0–1).",
    )
    forecast_mean:    Optional[List[float]] = Field(
        default=None,
        description="LSTM forecast mean (length H, 0–1). Zeros if omitted.",
    )
    forecast_std:     Optional[List[float]] = Field(
        default=None,
        description="LSTM forecast std  (length H, 0–1). Zeros if omitted.",
    )
    replicas_norm:    float = Field(
        ...,
        description="Current replicas / max_replicas (0–1).",
        ge=0.0, le=1.0,
    )
    cpu_per_pod_norm: float = Field(
        ...,
        description="Current cpu_per_pod / 1.0, clipped to [0,1].",
        ge=0.0, le=1.0,
    )


class ScaleResponse(BaseModel):
    """Response body for POST /scale-action."""
    action:       int   = Field(..., description="Discrete action index (0–4).")
    delta:        int   = Field(..., description="Replica delta: -2, -1, 0, +1, +2.")
    action_label: str   = Field(..., description="Human-readable action label.")
    agent_loaded: bool  = Field(..., description="Whether the PPO checkpoint was loaded.")


# ── /metrics ──────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    """Response body for GET /metrics."""
    total_reward:  Optional[float] = Field(None, description="Latest episode total reward.")
    slo_pct:       Optional[float] = Field(None, description="Latest episode SLO compliance %.")
    avg_replicas:  Optional[float] = Field(None, description="Latest episode mean replica count.")
    episode_count: int             = Field(0,    description="Total episodes run via /run-episode.")
    agent_loaded:  bool            = Field(...,  description="Whether PPO checkpoint is loaded.")
    model_loaded:  bool            = Field(...,  description="Whether LSTM checkpoint is loaded.")


# ── /config ───────────────────────────────────────────────────────────────────

class ConfigResponse(BaseModel):
    """Relevant configuration values exposed for dashboard consumption."""
    window_size:   int
    horizon:       int
    min_replicas:  int
    max_replicas:  int
    slo_threshold: float
    n_mc_samples:  int
    ci_lower_pct:  int
    ci_upper_pct:  int
