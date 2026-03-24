# ── Build stage: install dependencies ──────────────────────────────────────
FROM python:3.10-slim AS builder

# Install build tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use uv for fast, reproducible dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create virtual environment
RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies (cached layer — only re-runs when pyproject.toml changes)
COPY pyproject.toml ./
COPY src/ ./src/
RUN uv sync --no-dev


# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.10-slim

# ca-certificates — required for TLS connections to GCP / BigQuery
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built virtualenv from the builder
COPY --from=builder /app/.venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src/ ./src/
ENV PYTHONPATH=/app/src

# Dagster requires DAGSTER_HOME to be a writable directory
ENV DAGSTER_HOME=/opt/dagster/home
RUN mkdir -p /opt/dagster/home
