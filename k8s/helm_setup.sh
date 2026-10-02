#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# k8s/helm_setup.sh
# Install the kube-prometheus-stack via Helm for real CPU metrics.
#
# Prerequisites
#   - Helm 3.x installed  (https://helm.sh/docs/intro/install/)
#   - kubectl pointed at your Minikube / K8s cluster
#   - Minikube running: `minikube start --memory=4096 --cpus=2`
#
# Usage (run once):
#   bash k8s/helm_setup.sh
#
# After installation, port-forward Prometheus to verify:
#   kubectl port-forward svc/monitoring-kube-prometheus-prometheus -n monitoring 9090:9090
#   # then open http://localhost:9090 and query:
#   #   container_cpu_usage_seconds_total{pod=~"cloud-service-app-.*",container!=""}
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NAMESPACE="monitoring"
RELEASE="monitoring"

echo "→ Adding prometheus-community Helm repo ..."
helm repo add prometheus-community \
    https://prometheus-community.github.io/helm-charts
helm repo update

echo "→ Creating namespace '${NAMESPACE}' (if not exists) ..."
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "→ Installing kube-prometheus-stack (Grafana + Alertmanager disabled to save RAM) ..."
helm upgrade --install "${RELEASE}" \
    prometheus-community/kube-prometheus-stack \
    --namespace "${NAMESPACE}" \
    --set grafana.enabled=false \
    --set alertmanager.enabled=false \
    --set prometheus.prometheusSpec.resources.requests.memory=256Mi \
    --set prometheus.prometheusSpec.resources.requests.cpu=100m \
    --set prometheus.prometheusSpec.resources.limits.memory=512Mi \
    --set prometheus.prometheusSpec.resources.limits.cpu=300m \
    --set prometheus.prometheusSpec.retention=6h \
    --set prometheusOperator.resources.requests.memory=64Mi \
    --set prometheusOperator.resources.limits.memory=128Mi \
    --wait --timeout 5m

echo ""
echo "✓ kube-prometheus-stack installed successfully."
echo ""
echo "Port-forward Prometheus (run in a separate terminal):"
echo "  kubectl port-forward svc/${RELEASE}-kube-prometheus-prometheus -n ${NAMESPACE} 9090:9090"
echo ""
echo "Verify cAdvisor metrics are available:"
echo "  curl -s 'http://localhost:9090/api/v1/query?query=container_cpu_usage_seconds_total%7Bpod%3D~%22cloud-service-app-.*%22%2Ccontainer%21%3D%22%22%7D' | python -m json.tool | head -40"
