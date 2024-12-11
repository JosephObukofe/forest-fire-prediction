#!/bin/bash

set -e
set -u
set -o pipefail

echo "Starting Streamlit application..."
echo "Current directory: $(pwd)"

if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
  echo ".env file loaded successfully."
else
  echo "Warning: .env file not found. Proceeding without it."
fi

if [ ! -f /app/streamlit/app.py ]; then
  echo "Error: Streamlit app file '/app/streamlit/app.py' does not exist."
  exit 1
fi

if poetry run streamlit run /app/streamlit/app.py --server.address 0.0.0.0 --server.port 8501; then
  echo "Streamlit application started successfully."
else
  echo "Error: Failed to start Streamlit application."
  exit 1
fi