#!/bin/bash

source /Users/josephobukofe/predicting_forest_fires/.env

echo "Starting MinIO server..."
MINIO_ROOT_USER="$MINIO_ROOT_USER" MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" minio server /Users/josephobukofe/predicting_forest_fires/data/raw/minio \
    --address ":9278" &

echo "Starting Redis server..." 
redis-server \
    --port "$REDIS_PORT" \
    --requirepass "$REDIS_DB_PASSWORD" \
    --bind 127.0.0.1 &

echo "Starting MLflow..."
poetry run mlflow server \
    --backend-store-uri /Users/josephobukofe/predicting_forest_fires/mlruns \
    --default-artifact-root /Users/josephobukofe/predicting_forest_fires/mlartifacts \
    --host 127.0.0.1 \
    --port 5000 &

