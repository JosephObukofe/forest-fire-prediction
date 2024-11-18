#!/bin/bash

echo "Model evaluation initiated..."

if poetry run python evaluation.py; then
    echo "Model evaluation completed successfully!"
else
    echo "Model evaluation failed!" >&2
    exit 1
fi
