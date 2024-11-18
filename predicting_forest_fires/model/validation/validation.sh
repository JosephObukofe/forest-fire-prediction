#!/bin/bash

echo "Model validation initiated..."

if poetry run python validation.py; then
    echo "Model validation completed successfully!"
else
    echo "Model validation failed!" >&2
    exit 1
fi