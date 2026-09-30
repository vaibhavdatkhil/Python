"""
locust/locustfile.py
───────────────────
Real-Time Workload Generation with Locust for Kubernetes Autoscaling.

Simulates dynamic user traffic shaped by the Azure cloud workload trace.
Targets the deployed cloud microservice (`cloud-service-app`).

Usage:
  # Headless mode:
  locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 40 --spawn-rate 5

  # Web UI mode (interactive dashboard on http://localhost:8089):
  locust -f locust/locustfile.py --host http://localhost:8080
"""

import math
from pathlib import Path
from locust import HttpUser, task, between, LoadTestShape
import pandas as pd

TRACE_PATH = Path(__file__).resolve().parent.parent / "data" / "azure_vm_workload_trace.csv"

# Pre-load workload profile if trace file exists
_WORKLOAD_PROFILE = []
if TRACE_PATH.exists():
    try:
        _df = pd.read_csv(TRACE_PATH)
        if "cpu_util" in _df.columns:
            _WORKLOAD_PROFILE = _df["cpu_util"].tolist()
    except Exception:
        pass


class CloudServiceUser(HttpUser):
    """Simulates realistic end-user requests hitting the Kubernetes microservice."""
    wait_time = between(0.1, 0.5)

    @task(10)
    def index(self):
        """Standard web request to the root service endpoint."""
        self.client.get("/", name="GET / (Landing)")

    @task(3)
    def health_check(self):
        """Simulated health/status check."""
        self.client.get("/", name="GET /health (Ping)")


class AzureTraceLoadShape(LoadTestShape):
    """
    Dynamically adjusts user count over time to mimic real cloud traffic patterns
    sampled from the Azure VM trace.
    """
    time_limit = 1800  # 30 minutes total run
    step_duration = 15  # Change traffic level every 15 seconds
    min_users = 5
    max_users = 80
    spawn_rate = 10

    def tick(self):
        run_time = self.get_run_time()
        if run_time > self.time_limit:
            return None

        step_idx = int(run_time // self.step_duration)

        if _WORKLOAD_PROFILE and len(_WORKLOAD_PROFILE) > 0:
            trace_val = _WORKLOAD_PROFILE[step_idx % len(_WORKLOAD_PROFILE)]
            user_count = int(self.min_users + trace_val * (self.max_users - self.min_users))
        else:
            # Synthetic diurnal wave fallback
            wave = 0.5 + 0.5 * math.sin(2 * math.pi * run_time / 300)
            user_count = int(self.min_users + wave * (self.max_users - self.min_users))

        user_count = max(self.min_users, min(self.max_users, user_count))
        return (user_count, self.spawn_rate)
