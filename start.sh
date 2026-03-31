#!/usr/bin/env bash
# start.sh — starts the Flask backend for local development
# Usage: ./start.sh [--port 5000]
set -euo pipefail

PORT=${PORT:-5000}
VENV=".venv"

# ── Virtual env ────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ── Dependencies ───────────────────────────────────────────────────────────
echo "Installing dependencies..."
pip install --quiet --upgrade pip
# Install CPU-only torch first (much smaller download)
pip install --quiet torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install --quiet flask==3.0.3 flask-cors==4.0.1 gunicorn==22.0.0 numpy==1.26.4

# ── Launch ─────────────────────────────────────────────────────────────────
echo ""
echo "  TrafficRL backend starting on http://localhost:${PORT}"
echo "  Endpoints: GET /status  POST /train  POST /train/stop  POST /predict"
echo ""
export FLASK_ENV=development
python server.py --port "$PORT" 2>&1 | sed 's/^/[flask] /'
