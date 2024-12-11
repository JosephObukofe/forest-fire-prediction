#!/bin/bash

set -e
set -u
set -o pipefail

echo "Starting Celery worker and FastAPI application..."
echo "Current directory: $(pwd)"

if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
  echo ".env file loaded successfully."
else
  echo "Warning: .env file not found. Proceeding without it."
fi

cleanup() {
  echo "Shutting down Celery worker..."
  pkill -f "celery -A app.app_main.app.celery_app worker" || true
  echo "Cleanup complete. Exiting."
}
trap cleanup EXIT

echo "Running Celery worker..."
poetry run celery -A app.app_main.app.celery_app worker --loglevel=info &
CELERY_PID=$!

sleep 5

echo "Inspecting registered Celery tasks..."
if poetry run celery -A app.app_main.app.celery_app inspect registered; then
  echo "Celery tasks inspected successfully."
else
  echo "Warning: Failed to inspect registered Celery tasks."
fi

echo "Starting FastAPI application..."
poetry run uvicorn app.app_main.app:app --host 0.0.0.0 --port 8001
