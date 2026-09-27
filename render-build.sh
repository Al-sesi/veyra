#!/usr/bin/env bash
set -euo pipefail

echo "=== Veyra Render Build ==="

echo "Installing Python dependencies..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements-full.txt

echo "=== Build complete ==="