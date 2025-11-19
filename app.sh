#!/bin/bash

# Domino App Launcher for Diesel Fuel Quality Dashboard
# This script launches the Streamlit dashboard on the appropriate port for Domino

echo "Starting Enterprise Diesel Fuel Quality Management Dashboard..."

# Set the port for Domino Apps (8501 is standard for Streamlit)
export STREAMLIT_PORT=8501

# Ensure we're in the correct directory
cd /mnt/code

# Launch Streamlit with proper configuration for Domino
streamlit run scripts/diesel_dashboard.py \
    --server.port=${STREAMLIT_PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false