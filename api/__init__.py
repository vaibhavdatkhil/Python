"""
api/
─────
Phase 9: FastAPI REST backend for the K8s RL Autoscaling system.

Endpoints
─────────
  GET  /health          — liveness probe
  POST /forecast        — LSTM MC-Dropout forecast
  POST /scale-action    — PPO agent scaling decision
  GET  /metrics         — latest episode metrics
  GET  /config          — expose relevant config values
"""
