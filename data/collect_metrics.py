"""
data/collect_metrics.py
────────────────────────
Continuous live metrics collection from a Kubernetes cluster.

Polls Prometheus and the Kubernetes API every 15 seconds to collect:
  - CPU utilization (per-pod avg)
  - Memory utilization (per-pod avg)
  - Request rate (requests/sec from Prometheus http_requests_total)
  - P95 latency (from http_request_duration_seconds histogram)
  - Current replica count (from Kubernetes API)

Each row is timestamped and appended to ``data/live_metrics.csv`` immediately
after collection (flush on every write) so partial runs are always readable.

Prerequisites
─────────────
  - kube-prometheus-stack installed via Helm (see k8s/helm_setup.sh)
  - Prometheus port-forwarded: kubectl port-forward svc/monitoring-kube-prometheus-prometheus -n monitoring 9090:9090
  - Demo app deployed: kubectl apply -f k8s/demo_deployment.yaml

Usage
─────
  # Run for 5 minutes (20 samples at 15s intervals):
  python -m data.collect_metrics --duration 300

  # Run indefinitely (Ctrl-C to stop):
  python -m data.collect_metrics

  # Custom Prometheus URL:
  python -m data.collect_metrics --prometheus-url http://localhost:9090 --deployment cloud-service-app
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect-metrics")


# ── Prometheus Queries ────────────────────────────────────────────────────────

def query_prometheus(prometheus_url: str, query: str, timeout: float = 3.0) -> list[dict]:
    """Execute a PromQL query and return the result vector.

    Returns
    -------
    list of dicts with keys: metric (dict), value (list [timestamp, str_value])
    """
    try:
        url = f"{prometheus_url}/api/v1/query?" + urllib.parse.urlencode({"query": query})
        req = urllib.request.Request(url, headers={"User-Agent": "collect-metrics"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("status") == "success":
            return data.get("data", {}).get("result", [])
    except Exception as exc:
        logger.warning(f"Prometheus query failed: {exc}")
    return []


def get_cpu_util(prometheus_url: str, deployment: str) -> float | None:
    """Query average CPU utilization per pod (normalised to 0–1 range)."""
    query = (
        f'sum(rate(container_cpu_usage_seconds_total{{'
        f'pod=~"{deployment}-.*",container!=""}}'
        f'[1m])) by (pod)'
    )
    results = query_prometheus(prometheus_url, query)
    if not results:
        return None
    total_cpu = sum(float(r["value"][1]) for r in results)
    n_pods = len(results)
    # Normalise against 300m (0.3 cores) container limit from demo_deployment.yaml
    return (total_cpu / max(n_pods, 1)) / 0.3


def get_memory_util(prometheus_url: str, deployment: str) -> float | None:
    """Query average memory utilization per pod (normalised to 0–1 range)."""
    # container_memory_working_set_bytes is the metric K8s uses for OOMKill decisions
    query = (
        f'sum(container_memory_working_set_bytes{{'
        f'pod=~"{deployment}-.*",container!=""}}'
        f') by (pod)'
    )
    results = query_prometheus(prometheus_url, query)
    if not results:
        return None
    total_mem_bytes = sum(float(r["value"][1]) for r in results)
    n_pods = len(results)
    avg_mem_bytes = total_mem_bytes / max(n_pods, 1)
    # Normalise against 256Mi (268435456 bytes) container limit
    return avg_mem_bytes / 268_435_456.0


def get_request_rate(prometheus_url: str, deployment: str) -> float | None:
    """Query request rate (requests/sec) from http_requests_total counter."""
    # Assumes prometheus-fastapi-instrumentator exposes http_requests_total
    query = f'sum(rate(http_requests_total{{pod=~"{deployment}-.*"}}[1m]))'
    results = query_prometheus(prometheus_url, query)
    if results and len(results) > 0:
        return float(results[0]["value"][1])
    return None


def get_p95_latency(prometheus_url: str, deployment: str) -> float | None:
    """Query P95 latency (seconds) from http_request_duration_seconds histogram."""
    # histogram_quantile(0.95, ...) computes the 95th percentile
    query = (
        f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{'
        f'pod=~"{deployment}-.*"}}[1m])) by (le))'
    )
    results = query_prometheus(prometheus_url, query)
    if results and len(results) > 0:
        return float(results[0]["value"][1])
    return None


def get_replica_count(deployment: str, namespace: str = "default") -> int | None:
    """Query current replica count via kubectl."""
    try:
        cmd = [
            "kubectl", "get", "deployment", deployment,
            "-n", namespace,
            "-o", "jsonpath={.spec.replicas}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            return int(res.stdout.strip())
    except Exception as exc:
        logger.warning(f"kubectl get replicas failed: {exc}")
    return None


# ── Main Collection Loop ──────────────────────────────────────────────────────

def collect_loop(
    prometheus_url: str,
    deployment: str,
    namespace: str,
    output_path: Path,
    interval: float,
    duration: float | None,
) -> None:
    """Poll metrics every `interval` seconds, write to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine if we should write header (new file or empty file)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, "a", newline="") as f:
        fieldnames = [
            "timestamp",
            "cpu_util",
            "memory_util",
            "request_rate",
            "p95_latency_sec",
            "replicas",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        logger.info(f"Starting metrics collection → {output_path}")
        logger.info(f"  Prometheus    : {prometheus_url}")
        logger.info(f"  Deployment    : {deployment} (namespace: {namespace})")
        logger.info(f"  Interval      : {interval}s")
        logger.info(f"  Duration      : {duration if duration else 'indefinite (Ctrl-C to stop)'}")
        logger.info("")

        start_time = time.time()
        sample_count = 0

        try:
            while True:
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    break

                ts = datetime.utcnow().isoformat() + "Z"
                cpu = get_cpu_util(prometheus_url, deployment)
                mem = get_memory_util(prometheus_url, deployment)
                rps = get_request_rate(prometheus_url, deployment)
                p95 = get_p95_latency(prometheus_url, deployment)
                rep = get_replica_count(deployment, namespace)

                row = {
                    "timestamp": ts,
                    "cpu_util": f"{cpu:.4f}" if cpu is not None else "",
                    "memory_util": f"{mem:.4f}" if mem is not None else "",
                    "request_rate": f"{rps:.2f}" if rps is not None else "",
                    "p95_latency_sec": f"{p95:.4f}" if p95 is not None else "",
                    "replicas": rep if rep is not None else "",
                }

                writer.writerow(row)
                f.flush()
                sample_count += 1

                logger.info(
                    f"[{sample_count:4d}] CPU={cpu:.3f if cpu else 'N/A':>6}  "
                    f"Mem={mem:.3f if mem else 'N/A':>6}  "
                    f"RPS={rps:.1f if rps else 'N/A':>5}  "
                    f"P95={p95*1000:.1f if p95 else 'N/A':>5}ms  "
                    f"Replicas={rep if rep else 'N/A'}"
                )

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\nStopped by user (Ctrl-C).")

        logger.info(f"\nCollected {sample_count} samples → {output_path.resolve()}")


# ── CLI Entry-Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect live Kubernetes metrics from Prometheus and kubectl."
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="Prometheus server URL (default: http://localhost:9090).",
    )
    parser.add_argument(
        "--deployment",
        default="cloud-service-app",
        help="Deployment name to monitor (default: cloud-service-app).",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace (default: default).",
    )
    parser.add_argument(
        "--output",
        default="data/live_metrics.csv",
        help="Output CSV file path (default: data/live_metrics.csv).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Polling interval in seconds (default: 15).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Total duration in seconds. If omitted, runs indefinitely (stop with Ctrl-C).",
    )
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output

    collect_loop(
        prometheus_url=args.prometheus_url,
        deployment=args.deployment,
        namespace=args.namespace,
        output_path=output_path,
        interval=args.interval,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
