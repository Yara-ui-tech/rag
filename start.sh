#!/usr/bin/env bash
# ==============================================================================
# Production Startup Script for Linux / Cloud VPS
# ==============================================================================

set -e

PORT="${PORT:-5050}"
WORKERS="${WORKERS:-2}"
THREADS="${THREADS:-4}"

echo "======================================================================="
echo "        🚀 STARTING RAG TRAINING STUDIO (PRODUCTION) 🚀"
echo "======================================================================="
echo "PORT:    $PORT"
echo "WORKERS: $WORKERS"
echo "THREADS: $THREADS"
echo "======================================================================="

# Verify dependencies
pip install --no-cache-dir -q -r requirements.txt

# Launch via Gunicorn if available, otherwise fallback to server.py
if command -v gunicorn &> /dev/null; then
    exec gunicorn app:app \
        --bind "0.0.0.0:${PORT}" \
        --workers "${WORKERS}" \
        --threads "${THREADS}" \
        --timeout 120 \
        --access-logfile - \
        --error-logfile -
else
    exec python server.py
fi
