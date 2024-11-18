#!/bin/bash

echo "Model training initiated..."

if poetry run python training.py; then
    echo "Model training completed successfully!"
else
    echo "Model training failed!" >&2
    exit 1
fi