# Cancer Risk Assessment System - Docker Build
# ================================================
#
# Multi-stage Dockerfile that includes all trained models & data
#
# USAGE:
# ------
# 1. MOCK MODE (default - no GCP needed):
#    USE_MOCK=true docker-compose up
#
# 2. REAL VERTEX AI MODE (requires GCP credentials):
#    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json
#    docker-compose up
#
# INCLUDED:
# ---------
# ✓ Pre-trained ChromaDB vector store (161 chunks)
# ✓ Cached parsed NG12 PDF
# ✓ React/Vite frontend (pre-built)
# ✓ FastAPI backend
# ✓ All dependencies
#
# For other users to use their own GCP credentials:
# 1. Set environment variable: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/their/gcp-key.json
# 2. Or mount in docker-compose: volumes: [gcp-key.json:/app/secrets/gcp-key.json:ro]
# 3. Set GOOGLE_APPLICATION_CREDENTIALS env to point to mounted path
# 4. Start with: docker-compose up (without USE_MOCK)
#
# ================================================

# Multi-stage build: React frontend + Python backend
# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ .

# Build for production
RUN npm run build

# Stage 2: Python backend with all models
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p /app/data/ng12 \
    && mkdir -p /app/data/cache \
    && mkdir -p /app/vectorstore \
    && mkdir -p /app/logs

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Copy trained models and pre-processed data
# ============================================

# 1. ChromaDB vector store (trained embeddings)
COPY vectorstore/ ./vectorstore/

# 2. Cached parsed PDF model/data
COPY data/cache/ ./data/cache/

# 3. NG12 PDF guideline document
COPY data/ng12/ ./data/ng12/

# 4. Patient data
COPY data/patients.json ./data/

# Copy configuration and documentation files
COPY .env.docker .env
COPY README.md LICENSE prompts.md ./
COPY PROMPT_MANAGEMENT.md PROMPT_SYSTEM_SETUP.md DOCKER.md SETUP_GUIDE.md ./

# Copy frontend build artifacts
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create a simple static file server configuration for frontend
# This will be served by a lightweight HTTP server
RUN mkdir -p /app/static && \
    if [ -d /app/frontend/dist ]; then cp -r /app/frontend/dist/* /app/static/; fi

# Expose ports
# 8000: FastAPI backend
# 5173: Frontend dev server (optional)
# 3000: Frontend production server (optional)
EXPOSE 8000 5173 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Cancer Risk Assessment System..."\n\
echo "========================================="\n\
\n\
# Initialize services if needed\n\
if [ ! -f "/app/vectorstore/.initialized" ]; then\n\
  echo "First run: Initializing vector store..."\n\
  # Models are pre-loaded from COPY commands\n\
  touch /app/vectorstore/.initialized\n\
fi\n\
\n\
echo "Starting FastAPI backend on port 8000..."\n\
\n\
# Parse environment\n\
USE_MOCK=${USE_MOCK:-false}\n\
WORKERS=${WORKERS:-4}\n\
\n\
if [ "$USE_MOCK" = "true" ]; then\n\
  echo "Running in MOCK mode (no GCP credentials needed)"\n\
  USE_MOCK=true exec uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1\n\
else\n\
  echo "Running with full Vertex AI services"\n\
  exec uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers $WORKERS\n\
fi\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Build metadata
LABEL maintainer="Cancer Assessor Team" \
      description="Clinical Decision Support System for Cancer Risk Assessment" \
      version="1.0.0" \
      models="ChromaDB embeddings, Vertex AI Gemini, Marker PDF Parser"
