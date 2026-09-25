#!/usr/bin/env bash
set -euo pipefail

echo "=== Veyra Render Build ==="

echo "Installing ffmpeg..."
apt-get update -qq
apt-get install -y -qq --no-install-recommends ffmpeg
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "Installing Python dependencies..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "=== Build complete ==="