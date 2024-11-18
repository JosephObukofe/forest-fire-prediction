#!/bin/bash

echo "Data preprocessing initiated..."

if poetry run python processing.py; then
    echo "Data preprocessing completed successfully!"
    echo "Training and test sets persisted to Delta Lake."
else
    echo "Data preprocessing failed!" >&2
    exit 1
fi