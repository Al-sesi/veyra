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

COPY requirements.txt ./requirements.txt

# Install dependencies with caching
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchaudio==2.4.1 \
 && pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app (keep runtime slim).
COPY app/ ./app/
COPY runtime.txt ./runtime.txt

# Ensure HF cache dir + ledger dir exist (bound to persistent /data on HF Spaces)
RUN mkdir -p /data/hf-cache

EXPOSE 7860

CMD ["sh", "-c", "mkdir -p /data/hf-cache && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
