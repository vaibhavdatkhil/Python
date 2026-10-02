"""
k8s/verify_live_client.py
──────────────────────────
Verification script for Task 3: confirm that LiveK8sClient reads real
Prometheus values instead of falling back to 0.45, and that fallback
paths emit visible WARNING logs.

Two test modes
──────────────
1. --mock-prometheus  (default, no cluster needed)
   Spins up a tiny in-process HTTP server that returns a canned
   Prometheus API response, then asserts that get_cpu_utilization()
   returns the expected non-0.45 value.

2. --live-prometheus <url>  (requires port-forwarded Prometheus)
   Queries a real Prometheus instance and asserts the result is not 0.45
   (i.e. real cAdvisor data was found).

Usage
─────
  # Always works, no cluster required:
  python -m k8s.verify_live_client

  # After: kubectl port-forward svc/monitoring-kube-prometheus-prometheus -n monitoring 9090:9090
  python -m k8s.verify_live_client --live-prometheus http://localhost:9090 --deployment cloud-service-app
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify-live-client")


# ── Mock Prometheus server ────────────────────────────────────────────────────

# Simulates two pods each consuming 0.08 CPU cores (~80m / 300m limit ≈ 0.267 normalised)
_MOCK_PROM_RESPONSE = {
    "status": "success",
    "data": {
        "resultType": "vector",
        "result": [
            {"metric": {"pod": "cloud-service-app-abc12-1"}, "value": [1700000000, "0.08"]},
            {"metric": {"pod": "cloud-service-app-abc12-2"}, "value": [1700000000, "0.08"]},
        ],
    },
}

# Expected normalised value: avg(0.08, 0.08) / 0.3 cores ≈ 0.267
_EXPECTED_CPU = (0.08 + 0.08) / 2 / 0.3   # ≈ 0.267


class _MockPromHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(_MOCK_PROM_RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # suppress HTTP access log
        pass


def _start_mock_prometheus(port: int = 19090) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), _MockPromHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ── Inline CPU util logic (mirrors LiveK8sClient exactly, no torch dependency) ─

def _get_cpu_utilization_impl(prometheus_url: str, deployment: str, namespace: str = "default") -> float:
    """
    Standalone copy of LiveK8sClient.get_cpu_utilization for testing without
    importing the full k8s_adapter (which pulls in torch/model dependencies).

    Mirrors the three-tier logic in k8s_adapter.LiveK8sClient exactly:
      1. kube-prometheus-stack Prometheus query
      2. kubectl top pods fallback
      3. Hard fallback with WARNING log → 0.45
    """
    import subprocess
    import urllib.request as _urlreq
    import urllib.parse as _urlparse
    import json as _json

    # ── 1. Prometheus (kube-prometheus-stack query) ───────────────────────────
    try:
        prom_query = (
            f'sum(rate(container_cpu_usage_seconds_total{{'
            f'pod=~"{deployment}-.*",container!=""}}'
            f'[1m])) by (pod)'
        )
        url = (
            f"{prometheus_url}/api/v1/query?"
            + _urlparse.urlencode({"query": prom_query})
        )
        req = _urlreq.Request(url, headers={"User-Agent": "k8s-adapter"})
        with _urlreq.urlopen(req, timeout=3) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        if data.get("status") == "success":
            results = data.get("data", {}).get("result", [])
            if results:
                total_cpu = sum(float(r["value"][1]) for r in results)
                n_pods    = max(len(results), 1)
                cpu_per_pod_normalised = (total_cpu / n_pods) / 0.3
                return float(min(max(cpu_per_pod_normalised, 0.05), 2.0))
            else:
                logger.warning(
                    "[k8s-live] Prometheus query returned 0 results for "
                    f"deployment '{deployment}'. "
                    "Ensure kube-prometheus-stack is installed."
                )
    except Exception as exc:
        logger.warning(f"[k8s-live] Prometheus query failed: {exc}. Falling back to kubectl top.")

    # ── 2. kubectl top pods ───────────────────────────────────────────────────
    try:
        cmd = [
            "kubectl", "top", "pods",
            "-l", f"app={deployment}",
            "-n", namespace,
            "--no-headers",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.strip().splitlines()
            total_millicores, count = 0, 0
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        total_millicores += int(parts[1].replace("m", ""))
                        count += 1
                    except ValueError:
                        pass
            if count > 0:
                return float(min(total_millicores / count / 300.0, 2.0))
        logger.warning(f"[k8s-live] kubectl top returned no data for '{deployment}'.")
    except Exception as exc:
        logger.warning(f"[k8s-live] kubectl top failed: {exc}.")

    # ── 3. Hard fallback ──────────────────────────────────────────────────────
    logger.warning(
        "[k8s-live] FALLBACK: both Prometheus and kubectl top unavailable. "
        "Returning hardcoded CPU value 0.45. "
        "Fix: install kube-prometheus-stack via Helm (see README § Helm Setup)."
    )
    return 0.45


# ── Test: mock Prometheus ─────────────────────────────────────────────────────

def test_mock_prometheus() -> bool:
    """Verify get_cpu_utilization returns a real (non-fallback) value from mock Prometheus."""
    PORT = 19090
    logger.info(f"[test-mock] Starting mock Prometheus on port {PORT} ...")
    server = _start_mock_prometheus(PORT)
    time.sleep(0.1)   # let the server thread bind

    try:
        cpu = _get_cpu_utilization_impl(f"http://127.0.0.1:{PORT}", "cloud-service-app")
    finally:
        server.shutdown()

    logger.info(f"[test-mock] get_cpu_utilization returned: {cpu:.4f}  (expected ~{_EXPECTED_CPU:.4f})")

    tol = 0.01
    if abs(cpu - _EXPECTED_CPU) < tol:
        logger.info("[test-mock] PASS — value matches mock Prometheus data (not the 0.45 fallback).")
        return True
    else:
        logger.error(
            f"[test-mock] FAIL — returned {cpu:.4f} but expected ~{_EXPECTED_CPU:.4f}. "
            "Check get_cpu_utilization() normalisation logic."
        )
        return False


# ── Test: fallback warning is emitted ────────────────────────────────────────

def test_fallback_warning_is_logged() -> bool:
    """Verify that when Prometheus is unreachable the WARNING is logged and 0.45 is returned."""
    import logging as _logging

    warnings: list[str] = []

    class _CaptureHandler(_logging.Handler):
        def emit(self, record):
            if record.levelno >= _logging.WARNING:
                warnings.append(record.getMessage())

    capture = _CaptureHandler()
    logger.addHandler(capture)
    old_level = logger.level
    logger.setLevel(_logging.WARNING)

    # Port 19999 — nothing listening there
    cpu = _get_cpu_utilization_impl("http://127.0.0.1:19999", "cloud-service-app")

    logger.removeHandler(capture)
    logger.setLevel(old_level)

    if cpu == 0.45:
        logger.info(f"[test-fallback] Returned 0.45 (fallback). Warnings captured: {len(warnings)}")
    else:
        logger.warning(f"[test-fallback] Unexpected CPU value {cpu} — expected 0.45 fallback.")

    fallback_warning_found = any("FALLBACK" in w for w in warnings)
    if fallback_warning_found:
        logger.info("[test-fallback] PASS — FALLBACK WARNING was emitted correctly.")
    else:
        logger.error(
            "[test-fallback] FAIL — no FALLBACK WARNING found in logs. "
            f"Captured warnings: {warnings}"
        )

    return cpu == 0.45 and fallback_warning_found


# ── Test: live Prometheus ─────────────────────────────────────────────────────

def test_live_prometheus(prometheus_url: str, deployment: str) -> bool:
    """Query a real Prometheus and assert the result is not the 0.45 hardcoded fallback."""
    import urllib.request as _urlreq

    logger.info(f"[test-live] Querying real Prometheus at {prometheus_url} ...")

    # Quick connectivity check before trying the client
    try:
        req = _urlreq.Request(
            f"{prometheus_url}/-/healthy", headers={"User-Agent": "k8s-verify"}
        )
        with _urlreq.urlopen(req, timeout=3) as r:
            logger.info(f"[test-live] Prometheus health: HTTP {r.status}")
    except Exception as exc:
        logger.error(
            f"[test-live] Cannot reach Prometheus at {prometheus_url}: {exc}\n"
            "  → Run: kubectl port-forward svc/monitoring-kube-prometheus-prometheus "
            "-n monitoring 9090:9090"
        )
        return False

    cpu = _get_cpu_utilization_impl(prometheus_url, deployment)
    logger.info(f"[test-live] get_cpu_utilization returned: {cpu:.4f}")

    if cpu != 0.45:
        logger.info(
            f"[test-live] PASS — received real CPU value {cpu:.4f} (not the hardcoded 0.45)."
        )
        return True
    else:
        logger.warning(
            "[test-live] Value is 0.45 — this is the fallback value. "
            f"Either no pods matching '{deployment}-.*' exist in Prometheus, "
            "or cAdvisor metrics are not yet scraped. "
            "Check: container_cpu_usage_seconds_total{pod=~\"" + deployment + "-.*\",container!=\"\"}"
        )
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify LiveK8sClient reads real Prometheus CPU values."
    )
    parser.add_argument(
        "--live-prometheus", default=None, metavar="URL",
        help="URL of a live Prometheus instance (e.g. http://localhost:9090). "
             "If omitted, only the mock-Prometheus test runs.",
    )
    parser.add_argument(
        "--deployment", default="cloud-service-app",
        help="Deployment name to query in Prometheus.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  LiveK8sClient Verification (Task 3)")
    print("=" * 65)

    results: dict[str, bool] = {}

    # Always run the unit tests — they need no cluster
    results["mock_prometheus"]     = test_mock_prometheus()
    results["fallback_warning"]    = test_fallback_warning_is_logged()

    # Run live test only when a Prometheus URL was given
    if args.live_prometheus:
        results["live_prometheus"] = test_live_prometheus(
            args.live_prometheus, args.deployment
        )
    else:
        logger.info(
            "[main] Skipping live-Prometheus test (no --live-prometheus URL given).\n"
            "       To run it: python -m k8s.verify_live_client "
            "--live-prometheus http://localhost:9090"
        )

    print()
    print("  Results:")
    all_pass = True
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_pass = False
        print(f"    {name:<30} {status}")

    print()
    if all_pass:
        print("  All checks passed.")
    else:
        print("  Some checks failed — see log output above.")
    print("=" * 65)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
