#!/bin/bash

# Heimdall Quick Start Script
# This script helps you get Heimdall running quickly

echo "🔒 Heimdall Quick Start"
echo "======================"
echo ""

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    echo "📝 Setting up .env file..."
    if [ -f "backend/.env.example" ]; then
        cp backend/.env.example backend/.env
        echo "✅ Created backend/.env from .env.example"
        echo ""
        echo "⚠️  IMPORTANT: Edit backend/.env and add your XAI_API_KEY!"
        echo "   Get your key from: https://console.x.ai/"
        echo ""
        read -p "Press Enter after you've added your XAI_API_KEY to backend/.env..."
    else
        echo "❌ backend/.env.example not found"
        exit 1
    fi
else
    echo "✅ backend/.env already exists"
fi

# Check if XAI_API_KEY is set
if grep -q "XAI_API_KEY=your_xai_api_key_here" backend/.env 2>/dev/null || ! grep -q "XAI_API_KEY=" backend/.env 2>/dev/null; then
    echo "⚠️  Warning: XAI_API_KEY not configured in backend/.env"
    echo "   Please edit backend/.env and set your XAI_API_KEY"
    echo ""
fi

echo ""
echo "🚀 Starting Heimdall..."
echo ""
echo "This will start both backend and frontend."
echo "Press Ctrl+C to stop both services."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend on http://localhost:8000..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
else
    source venv/bin/activate
fi

# Load .env
export $(grep -v '^#' .env | xargs)

uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 3

# Start frontend
FRONTEND_PORT=3001
echo "Starting frontend on http://localhost:${FRONTEND_PORT}..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install --silent
fi

PORT=${FRONTEND_PORT} npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Heimdall is running!"
echo ""
echo "📱 Frontend: http://localhost:${FRONTEND_PORT}"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for both processes
wait

