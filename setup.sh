#!/bin/bash

# Heimdall Setup Script
echo "🔒 Setting up Heimdall..."

# Create agents directory
mkdir -p agents

# Clone Hound repository
if [ ! -d "agents/hound" ]; then
    echo "📦 Cloning Hound repository..."
    git clone https://github.com/scabench-org/hound.git agents/hound
else
    echo "✓ Hound already cloned"
fi

# Clone BRAMA repository
if [ ! -d "agents/brama" ]; then
    echo "📦 Cloning BRAMA repository..."
    git clone https://github.com/oborys/security-ai-agent-brama.git agents/brama
else
    echo "✓ BRAMA already cloned"
fi

# Setup Python backend
echo "🐍 Setting up Python backend..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Setup Node.js frontend
echo "⚛️  Setting up Next.js frontend..."
cd frontend
npm install
cd ..

echo "✅ Setup complete!"
echo ""
echo "To start the backend:"
echo "  cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo ""
echo "To start the frontend:"
echo "  cd frontend && npm run dev"
echo ""
echo "Then open http://localhost:3000 in your browser"

