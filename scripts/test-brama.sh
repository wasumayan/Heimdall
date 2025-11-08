#!/bin/bash

# Test BRAMA red-team scanner
echo "🔍 Testing BRAMA Red-Team Scanner..."
echo ""

# Check if URL provided
if [ -z "$1" ]; then
    echo "Usage: ./test-brama.sh <url>"
    echo "Example: ./test-brama.sh https://example.com"
    exit 1
fi

URL=$1

echo "Scanning: $URL"
echo ""

# Set API key if provided
if [ -n "$XAI_API_KEY" ]; then
    export XAI_API_KEY
fi

# Run BRAMA scan
cd "$(dirname "$0")/../agents/brama"

if [ ! -f "scan_url.py" ]; then
    echo "❌ Error: scan_url.py not found"
    exit 1
fi

echo "Running scan..."
python3 scan_url.py "$URL" | python3 -m json.tool

echo ""
echo "✅ Scan complete!"

