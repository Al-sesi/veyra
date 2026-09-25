# syntax=docker/dockerfile:1.6

FROM python:3.11-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-cache \
    TRANSFORMERS_CACHE=/data/hf-cache \
    VEYRA_DB_PATH=/data/ledger.db \
    PORT=7860

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
        tzdata \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY requirements-full.txt ./requirements-full.txt

# Install PyTorch CPU wheel first (index URL in requirements-full.txt), then the rest.
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements-full.txt

# Copy the FastAPI app (keep runtime slim).
COPY app/ ./app/
COPY runtime.txt ./runtime.txt 2>/dev/null || true

# Ensure HF cache dir + ledger dir exist (bound to persistent /data on HF Spaces)
RUN mkdir -p /data/hf-cache

EXPOSE 7860

CMD ["sh", "-c", "mkdir -p /data/hf-cache && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
