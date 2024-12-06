#!/bin/bash

echo "Starting Streamlit application..."
echo "Current directory: $(pwd)"

if [ -f /app/streamlit/.env ]; then
  set -o allexport
  source /app/streamlit/.env
  set +o allexport
  echo ".env file loaded successfully."
else
  echo "Warning: .env file not found. Proceeding without it."
fi

if poetry run streamlit run streamlit/app.py; then
  echo "Streamlit application started successfully."
else
  echo "Error: Failed to start Streamlit application."
  exit 1
fi