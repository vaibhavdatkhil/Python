# ── Stage: runtime ───────────────────────────────────────────────────────────
# Lightweight Python 3.11 slim image — keeps the container small enough
# for a resource-limited laptop (8 GB RAM) running Minikube.
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered stdout/stderr (better container logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (layer-caches well between code changes)
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    prometheus-fastapi-instrumentator==6.1.0

# Copy application source
COPY main.py .

# Expose the service port
EXPOSE 8000

# Run uvicorn; bind to all interfaces so Kubernetes can reach it
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
