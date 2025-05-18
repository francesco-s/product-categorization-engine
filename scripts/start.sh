#!/bin/bash
# scripts/start.sh

set -e

echo "Starting Product Categorization API"

# Set default environment variables if not already set
export MODEL_PATH=${MODEL_PATH:-/app/data/models/best_model.pt}
export LOG_LEVEL=${LOG_LEVEL:-info}
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}
export WORKERS=${WORKERS:-1}

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# Start the API
echo "Running API with model: $MODEL_PATH"
echo "Log level: $LOG_LEVEL"
echo "Listening on: $HOST:$PORT"

exec uvicorn src.api.app:app --host $HOST --port $PORT --workers $WORKERS --log-level $LOG_LEVEL