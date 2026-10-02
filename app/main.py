"""
app/main.py
───────────
Demo FastAPI microservice for Kubernetes autoscaling experiments.

Endpoints
─────────
  GET /         — welcome message
  GET /health   — liveness probe (always 200)
  GET /work     — CPU-bound task (~200,000 sqrt iterations) so Prometheus
                  metrics show real CPU utilisation under load
  GET /metrics  — Prometheus-format metrics (instrumented automatically)

Prometheus Instrumentation
──────────────────────────
  Uses prometheus-fastapi-instrumentator to auto-expose:
    - http_requests_total (counter, labelled by method / handler / status)
    - http_request_duration_seconds (histogram, latency)

Build & run locally
───────────────────
  pip install fastapi uvicorn[standard] prometheus-fastapi-instrumentator
  uvicorn app.main:app --reload --port 8000

Docker
──────
  eval $(minikube docker-env)
  docker build -t demo-app:1 app/
  kubectl apply -f k8s/demo_deployment.yaml
"""

from __future__ import annotations

import math
import os
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="K8s RL Autoscaler — Demo Microservice",
    description="CPU-bound endpoint for realistic autoscaling experiments.",
    version="1.0.0",
)

# ── Prometheus instrumentation ────────────────────────────────────────────────
# Exposes /metrics in Prometheus text format (request count + latency histogram)
Instrumentator().instrument(app).expose(app)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Welcome", tags=["info"])
def root():
    """Service root — basic hello."""
    return {"service": "demo-app", "version": "1.0.0", "status": "running"}


@app.get("/health", summary="Liveness probe", tags=["info"])
def health():
    """Kubernetes liveness / readiness probe. Always returns 200."""
    return {"status": "ok"}


@app.get("/work", summary="CPU-bound work endpoint", tags=["load"])
def work(iterations: int = 200_000):
    """
    Performs a CPU-bound loop so that Prometheus CPU metrics reflect real load.

    Parameters
    ----------
    iterations : number of sqrt computations (default 200,000).
                 Increase via ?iterations=N for heavier load in tests.

    Returns
    -------
    JSON with result checksum, duration_ms, and iteration count.
    """
    start = time.perf_counter()
    # Tight arithmetic loop — intentionally CPU-bound
    acc = 0.0
    for i in range(1, iterations + 1):
        acc += math.sqrt(i)

    duration_ms = (time.perf_counter() - start) * 1000.0
    return {
        "iterations": iterations,
        "checksum":   round(acc % 1e6, 4),   # partial result to prevent DCE
        "duration_ms": round(duration_ms, 2),
        "status": "ok",
    }


# ── Entry-point (for direct `python app/main.py` invocation) ──────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
