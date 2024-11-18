#!/bin/bash

echo "Running celery worker..."
poetry run celery \
    -A fastapi.app.celery_app worker \
    --loglevel=info &

poetry run celery \
    -A fastapi.app.celery_app inspect registered &

echo "Starting FastAPI application..."
poetry run uvicorn fastapi.app:app \
    --reload \
    --host 0.0.0.0 \
    --port 8001 

wait

