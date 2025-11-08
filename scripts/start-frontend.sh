#!/bin/bash

# Start Heimdall Frontend
cd "$(dirname "$0")/../frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "Starting Heimdall Frontend on http://localhost:3000"
npm run dev

