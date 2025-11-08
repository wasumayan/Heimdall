#!/bin/bash

# Start Heimdall Backend
cd "$(dirname "$0")/../backend"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "venv/bin/activate" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting Heimdall Backend on http://localhost:8000"
uvicorn main:app --reload --host 0.0.0.0 --port 8000

