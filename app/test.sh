#!/bin/bash

source /Users/josephobukofe/predicting_forest_fires/.env

echo "Starting Redis CLI..."
redis-cli -h "$REDIS_HOST_NAME" -p "$REDIS_PORT" -a "$REDIS_DB_PASSWORD"