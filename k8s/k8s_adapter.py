"""
k8s/k8s_adapter.py
──────────────────
Kubernetes Adapter & Intelligent Autoscaling Controller.

Connects the trained LSTM forecaster and PPO reinforcement learning agent
to a Kubernetes cluster (or a high-fidelity local Mock Cluster).

Supported Execution Modes
─────────────────────────
  1. mock : Fully simulated Kubernetes Pod lifecycle. Perfect for demos,
            testing, and environments without Docker / Minikube.
  2. live : Directly queries and scales a real Kubernetes cluster using
            `kubectl` or the official Kubernetes Python API.

Usage
─────
  # Run 30 steps of mock autoscaling against simulated cloud workload:
  python -m k8s.k8s_adapter --mode mock --steps 30

  # Run live against a real Minikube / K8s cluster:
  python -m k8s.k8s_adapter --mode live --deployment cloud-service-app --namespace default
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_trace
from model.inference import load_checkpoint as load_lstm_checkpoint, mc_predict
from rl_agent.ppo import load_ppo_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("k8s-autoscaler")


# ── Cluster Interface ─────────────────────────────────────────────────────────

class K8sClusterInterface(ABC):
    """Abstract interface for Kubernetes cluster operations."""

    @abstractmethod
    def get_replicas(self, deployment: str, namespace: str = "default") -> int:
        """Get current replica count for the deployment."""
        pass

    @abstractmethod
    def scale(self, deployment: str, replicas: int, namespace: str = "default") -> bool:
        """Scale deployment to the specified replica count."""
        pass

    @abstractmethod
    def get_cpu_utilization(self, deployment: str, namespace: str = "default") -> float:
        """Get the current average CPU utilization per pod (0.0 to 1.0+)."""
        pass


# ── Mock Kubernetes Cluster ───────────────────────────────────────────────────

class MockK8sCluster(K8sClusterInterface):
    """High-fidelity simulated Kubernetes cluster.

    Simulates:
    - Pod startup lag (pods transition Pending -> Running over scale_lag_steps)
    - Resource saturation when pods are under-provisioned
    - Audit events resembling real Kubernetes Events
    """

    def __init__(
        self,
        initial_replicas: int = 3,
        min_replicas: int = 1,
        max_replicas: int = 10,
        lag_steps: int = 2,
    ):
        self.active_replicas = initial_replicas
        self.target_replicas = initial_replicas
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.lag_steps = lag_steps
        self.pending_actions: list[tuple[int, int]] = []  # (target_r, steps_left)
        self.last_workload: float = 0.50
        self.step_counter: int = 0

    def get_replicas(self, deployment: str, namespace: str = "default") -> int:
        return self.active_replicas

    def scale(self, deployment: str, replicas: int, namespace: str = "default") -> bool:
        replicas = int(np.clip(replicas, self.min_replicas, self.max_replicas))
        prev = self.active_replicas
        delta = replicas - prev

        if delta != 0:
            action_type = "Scaled up" if delta > 0 else "Scaled down"
            logger.info(
                f"[k8s-event] Normal ScalingReplicaSet: {action_type} replica set "
                f"{deployment}-5f89c from {prev} to {replicas} (lag: {self.lag_steps} steps)"
            )
            # Queue the action with lag
            self.pending_actions.append((replicas, self.lag_steps))
            self.target_replicas = replicas
            return True
        return False

    def advance_time(self, workload_cpu: float):
        """Simulate one step forward in cluster time."""
        self.step_counter += 1
        self.last_workload = workload_cpu

        # Process pending scaling lag queue
        new_pending = []
        for target, remaining in self.pending_actions:
            if remaining <= 1:
                old = self.active_replicas
                self.active_replicas = target
                logger.info(
                    f"[k8s-event] Normal SuccessfulCreate: Pods ready. Active replicas updated: {old} -> {target}"
                )
            else:
                new_pending.append((target, remaining - 1))
        self.pending_actions = new_pending

    def get_cpu_utilization(self, deployment: str, namespace: str = "default") -> float:
        """Simulate per-pod CPU utilization based on current workload and active pods."""
        effective_replicas = max(self.active_replicas, 1)
        # Total cluster CPU demand divided across active replicas
        cpu_per_pod = (self.last_workload * 2.5) / effective_replicas
        return float(cpu_per_pod)


# ── Live Kubernetes Client (via kubectl or python-client) ─────────────────────

class LiveK8sClient(K8sClusterInterface):
    """Connects directly to an active Kubernetes cluster using kubectl CLI and Prometheus."""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        # Verify kubectl connection
        try:
            res = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode != 0:
                raise ConnectionError(f"kubectl cannot reach cluster: {res.stderr.strip()}")
            logger.info("[k8s-live] Connected to active Kubernetes cluster.")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize LiveK8sClient: {exc}")

    def get_replicas(self, deployment: str, namespace: str = "default") -> int:
        cmd = [
            "kubectl", "get", "deployment", deployment,
            "-n", namespace,
            "-o", "jsonpath={.spec.replicas}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.warning(f"Failed to get replicas for {deployment}: {res.stderr}")
            return 1
        return int(res.stdout.strip())

    def scale(self, deployment: str, replicas: int, namespace: str = "default") -> bool:
        cmd = [
            "kubectl", "scale", f"deployment/{deployment}",
            f"--replicas={replicas}",
            "-n", namespace,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info(f"[k8s-live] Executed: kubectl scale deployment/{deployment} --replicas={replicas}")
            return True
        else:
            logger.error(f"[k8s-live] Scale command failed: {res.stderr}")
            return False

    def get_cpu_utilization(self, deployment: str, namespace: str = "default") -> float:
        """Fetch average CPU utilization per pod from Prometheus or kubectl top fallback."""
        # 1. Query Prometheus metrics if accessible
        try:
            import urllib.request
            import urllib.parse
            import json

            prom_query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{deployment}-.*"}}[1m]))'
            url = f"{self.prometheus_url}/api/v1/query?" + urllib.parse.urlencode({"query": prom_query})
            req = urllib.request.Request(url, headers={"User-Agent": "k8s-adapter"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "success":
                    results = data.get("data", {}).get("result", [])
                    if results and len(results) > 0:
                        total_cpu = float(results[0]["value"][1])
                        replicas = max(self.get_replicas(deployment, namespace), 1)
                        return min(max((total_cpu / replicas) / 0.5, 0.05), 1.0)
        except Exception:
            pass

        # 2. Try kubectl top pods fallback
        try:
            cmd = ["kubectl", "top", "pods", "-l", f"app={deployment}", "-n", namespace, "--no-headers"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().splitlines()
                total_millicores = 0
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        val = parts[1].replace("m", "")
                        total_millicores += int(val)
                avg_milli = total_millicores / max(len(lines), 1)
                return avg_milli / 500.0
        except Exception:
            pass

        # 3. Graceful default when metrics server is initializing
        return 0.45


# ── Autoscaling Controller ────────────────────────────────────────────────────

class RLAutoscalingController:
    """Intelligent Autoscaling Controller driven by LSTM + PPO.

    Includes safety guardrails:
      - Minimum and maximum replica bounds.
      - Actuation cooldown to prevent thrashing.
      - Uncertainty-aware fallback (reverts to reactive HPA if forecast variance is extreme).
    """

    ACTION_MAP = {0: -2, 1: -1, 2: 0, 3: +1, 4: +2}

    def __init__(
        self,
        config_path: Path | str = PROJECT_ROOT / "config.yaml",
        cluster: K8sClusterInterface | None = None,
        deployment: str = "cloud-service-app",
        namespace: str = "default",
        cooldown_steps: int = 1,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.cluster = cluster or MockK8sCluster()
        self.deployment = deployment
        self.namespace = namespace
        self.cooldown_steps = cooldown_steps
        self.last_action_step = -100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        lstm_ckpt = PROJECT_ROOT / "checkpoints" / "best_model.pt"
        ppo_ckpt = PROJECT_ROOT / "checkpoints" / "ppo_agent.pt"
        if not ppo_ckpt.exists():
            ppo_ckpt = PROJECT_ROOT / "checkpoints" / "ppo_agent_final.pt"

        self.lstm_model, self.scaler, _ = load_lstm_checkpoint(lstm_ckpt, device=self.device)
        self.ppo_policy, _ = load_ppo_checkpoint(ppo_ckpt, device=self.device)
        logger.info(f"Loaded LSTM checkpoint from {lstm_ckpt.name}")
        logger.info(f"Loaded PPO policy checkpoint from {ppo_ckpt.name}")

        self.W = self.cfg["preprocessing"]["window_size"]
        self.H = self.cfg["preprocessing"]["horizon"]
        self.min_r = self.cfg["rl_env"]["min_replicas"]
        self.max_r = self.cfg["rl_env"]["max_replicas"]
        self.slo_thr = self.cfg["rl_env"]["slo_threshold"]

    def build_observation(
        self,
        cpu_history: np.ndarray,
        current_replicas: int,
        cpu_per_pod: float,
    ) -> tuple[np.ndarray, float]:
        """Construct the 5-component observation vector required by the PPO policy."""
        # 1. CPU look-back window (scaled [0, 1])
        scaled_history = self.scaler.transform(cpu_history.reshape(-1, 1)).ravel()

        # 2. LSTM MC-Dropout forecast
        mean_fc, lower_fc, upper_fc = mc_predict(
            self.lstm_model,
            scaled_history,
            n_samples=30,
            scaler=self.scaler,
            device=self.device,
        )
        # Scaled forecast mean & std
        mean_scaled = self.scaler.transform(mean_fc.reshape(-1, 1)).ravel()
        std_scaled = (upper_fc - lower_fc) / (2.0 * 1.645)  # approx std in original scale
        std_scaled = std_scaled / (self.scaler.data_range_[0] + 1e-8)

        # 3. Normalised replica count [0, 1]
        rep_norm = (current_replicas - self.min_r) / max(self.max_r - self.min_r, 1)

        # 4. Normalised CPU per pod
        cpp_norm = min(cpu_per_pod / (self.slo_thr * 1.5), 2.0)

        # Combined observation vector
        obs = np.concatenate([
            scaled_history,          # W steps
            mean_scaled,             # H steps
            std_scaled,              # H steps
            np.array([rep_norm], dtype=np.float32),
            np.array([cpp_norm], dtype=np.float32),
        ]).astype(np.float32)

        uncertainty_score = float(np.mean(std_scaled))
        return obs, uncertainty_score

    def decide_and_scale(
        self,
        cpu_history: np.ndarray,
        step_idx: int,
    ) -> dict:
        """Run one control loop tick: observe, forecast, decide, and actuate."""
        current_replicas = self.cluster.get_replicas(self.deployment, self.namespace)
        cpu_per_pod = self.cluster.get_cpu_utilization(self.deployment, self.namespace)

        obs_np, uncertainty = self.build_observation(cpu_history, current_replicas, cpu_per_pod)
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

        # Run policy inference
        with torch.no_grad():
            action_idx_t, _, _, _ = self.ppo_policy.act(obs_t)
            action_idx = int(action_idx_t.item())

        delta_r = self.ACTION_MAP.get(action_idx, 0)
        slo_met = cpu_per_pod <= self.slo_thr

        # Safety Fallback: if uncertainty is extreme, fall back to reactive threshold
        fallback_active = False
        if uncertainty > 0.45:
            fallback_active = True
            if cpu_per_pod > self.slo_thr:
                delta_r = +1
            elif cpu_per_pod < self.slo_thr * 0.4:
                delta_r = -1
            else:
                delta_r = 0

        # Cooldown guard: suppress rapid oscillating changes
        if delta_r != 0 and (step_idx - self.last_action_step) < self.cooldown_steps:
            delta_r = 0

        target_replicas = int(np.clip(current_replicas + delta_r, self.min_r, self.max_r))

        if target_replicas != current_replicas:
            self.cluster.scale(self.deployment, target_replicas, self.namespace)
            self.last_action_step = step_idx

        return {
            "step": step_idx,
            "current_replicas": current_replicas,
            "target_replicas": target_replicas,
            "delta": delta_r,
            "cpu_per_pod": round(cpu_per_pod, 3),
            "slo_met": slo_met,
            "uncertainty": round(uncertainty, 4),
            "fallback": fallback_active,
        }


# ── CLI Runner ────────────────────────────────────────────────────────────────

def run_controller_demo(
    mode: str = "mock",
    steps: int = 30,
    deployment: str = "cloud-service-app",
    namespace: str = "default",
):
    logger.info(f"Starting K8s RL Autoscaler Controller in '{mode}' mode for deployment '{deployment}'...")

    # Load real benchmark trace for driving the simulation
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Use Azure benchmark trace if available, else standard loader
    trace_path = PROJECT_ROOT / "data" / "azure_vm_workload_trace.csv"
    if trace_path.exists():
        df_trace = pd.read_csv(trace_path, parse_dates=["timestamp"])
        logger.info(f"Loaded Azure benchmark trace from {trace_path.name} ({len(df_trace)} timesteps)")
    else:
        df_trace = load_trace(cfg=cfg)

    cpu_values = df_trace["cpu_util"].to_numpy()
    W = cfg["preprocessing"]["window_size"]

    if mode == "live":
        try:
            cluster = LiveK8sClient()
        except Exception as e:
            logger.error(f"Cannot initialize live K8s client: {e}. Falling back to mock mode.")
            cluster = MockK8sCluster()
    else:
        cluster = MockK8sCluster(
            initial_replicas=3,
            min_replicas=cfg["rl_env"]["min_replicas"],
            max_replicas=cfg["rl_env"]["max_replicas"],
            lag_steps=cfg["rl_env"]["scale_lag_steps"],
        )

    controller = RLAutoscalingController(
        cluster=cluster,
        deployment=deployment,
        namespace=namespace,
    )

    logger.info(f"{'STEP':<6} {'WORKLOAD':<10} {'REPLICAS':<10} {'CPU/POD':<10} {'ACTION':<10} {'SLO':<8}")
    logger.info("-" * 60)

    for i in range(steps):
        # Workload history for the current window
        history_window = cpu_values[i : i + W]
        cur_cpu = cpu_values[i + W]

        # Advance mock cluster time if applicable
        if isinstance(cluster, MockK8sCluster):
            cluster.advance_time(cur_cpu)

        res = controller.decide_and_scale(history_window, step_idx=i)
        action_sym = f"{res['delta']:+d}" if res['delta'] != 0 else "0 (Hold)"
        slo_str = "MET" if res["slo_met"] else "VIOLATED"

        logger.info(
            f"{i+1:<6} {cur_cpu:<10.3f} {res['target_replicas']:<10} "
            f"{res['cpu_per_pod']:<10.3f} {action_sym:<10} {slo_str:<8}"
        )

    logger.info("-" * 60)
    logger.info("[OK] Controller execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kubernetes RL Autoscaling Adapter")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock", help="Execution mode (mock or live K8s cluster)")
    parser.add_argument("--steps", type=int, default=25, help="Number of control loop steps to simulate")
    parser.add_argument("--deployment", type=str, default="cloud-service-app", help="Target K8s deployment name")
    parser.add_argument("--namespace", type=str, default="default", help="Target K8s namespace")
    args = parser.parse_args()

    run_controller_demo(
        mode=args.mode,
        steps=args.steps,
        deployment=args.deployment,
        namespace=args.namespace,
    )
