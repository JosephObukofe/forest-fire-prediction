#!/bin/bash

echo "Starting Celery worker and FastAPI application..."
echo "Current directory: $(pwd)"

if [ -f /app/.env ]; then
  set -o allexport
  source /app/.env
  set +o allexport
  echo ".env file loaded successfully."
else
  echo "Warning: .env file not found. Proceeding without it."
fi

echo "Running Celery worker..."
if poetry run celery -A app.app_main.app.celery_app worker --loglevel=info & then
  echo "Celery worker started successfully."
else
  echo "Error: Failed to start Celery worker."
  exit 1
fi

echo "Inspecting registered Celery tasks..."
if poetry run celery -A app.app_main.app.celery_app inspect registered & then
  echo "Celery tasks inspected successfully."
else
  echo "Warning: Failed to inspect registered Celery tasks."
fi

echo "Starting FastAPI application..."
if poetry run uvicorn app.app_main.app:app --host 0.0.0.0 --port 8001; then
  echo "FastAPI application started successfully."
else
  echo "Error: Failed to start FastAPI application."
  exit 1
fi

