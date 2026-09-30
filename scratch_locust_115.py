"""
locust/locustfile.py
────────────────────
Real-Time Workload Replay using Locust.

Shapes HTTP traffic to match the cloud workload trace (Azure / Alibaba / Synthetic).
Implements a custom LoadTestShape so that active users and request rates scale
dynamically over time, mirroring realistic traffic patterns against Kubernetes pods.

Usage:
──────
  # 1. Install locust:
  pip install locust

  # 2. Run with web UI (opens at http://localhost:8089):
  locust -f locust/locustfile.py --host http://localhost:80

  # 3. Or run headless (auto-replaying the trace):
  locust -f locust/locustfile.py --host http://localhost:80 --headless
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from locust import HttpUser, task, between, LoadTestShape

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_trace


class MicroserviceUser(HttpUser):
    """Simulates realistic user traffic against the microservice."""
    wait_time = between(0.1, 0.5)

    @task(3)
    def index_page(self):
        """Standard HTTP GET request to test service throughput."""
        self.client.get("/", name="GET / (Homepage)")

    @task(1)
    def health_check(self):
        """Simulate health probes / API status checks."""
        self.client.get("/", name="GET /health (Ping)")


class TraceShapedTraffic(LoadTestShape):
    """Replays the time-series workload trace to dynamically adjust user count.

    Every tick (second), this shape determines the number of concurrent Locust users
    proportional to the trace CPU utilization curve.
    """

    # Scaling configuration
    max_users = 80             # Maximum concurrent users at peak (1.0 CPU)
    min_users = 5              # Minimum baseline users at trough (0.05 CPU)
    step_duration = 30         # Seconds to hold each trace step (accelerated replay)

    def __init__(self):
        super().__init__()
        # Load the trace (Azure benchmark or synthetic depending on config.yaml)
        azure_path = PROJECT_ROOT / "data" / "azure_vm_workload_trace.csv"
        if azure_path.exists():
            df = pd.read_csv(azure_path)
            self.trace_values = df["cpu_util"].to_numpy()
        else:
            df = load_trace()
            self.trace_values = df["cpu_util"].to_numpy()

    def tick(self) -> tuple[int, float] | None:
        """Called every second to compute current user target and spawn rate."""
        run_time = self.get_run_time()

        # Determine which timestep of the trace we are currently replaying
        step_idx = int(run_time // self.step_duration) % len(self.trace_values)
        current_cpu_factor = float(self.trace_values[step_idx])

        # Target users calculated linearly from trace utilization factor
        target_users = int(
            self.min_users + current_cpu_factor * (self.max_users - self.min_users)
        )

        # Spawn rate (how fast users scale up/down per second)
        spawn_rate = 5.0

        return (target_users, spawn_rate)
