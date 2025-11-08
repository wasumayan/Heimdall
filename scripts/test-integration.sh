#!/bin/bash

# Test script to verify Heimdall integration

echo "🔍 Testing Heimdall Integration..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check directory structure
echo "1. Checking directory structure..."
if [ -d "agents/hound" ] || [ -d "hound" ]; then
    echo -e "${GREEN}✓ Hound found${NC}"
else
    echo -e "${RED}✗ Hound not found${NC}"
fi

if [ -d "agents/brama" ]; then
    echo -e "${GREEN}✓ BRAMA found${NC}"
else
    echo -e "${RED}✗ BRAMA not found${NC}"
fi

if [ -d "backend" ]; then
    echo -e "${GREEN}✓ Backend directory found${NC}"
else
    echo -e "${RED}✗ Backend directory not found${NC}"
fi

if [ -d "frontend" ]; then
    echo -e "${GREEN}✓ Frontend directory found${NC}"
else
    echo -e "${RED}✗ Frontend directory not found${NC}"
fi

echo ""

# Test 2: Check virtual environments
echo "2. Checking virtual environments..."

if [ -d "backend/venv" ]; then
    echo -e "${GREEN}✓ Backend venv exists${NC}"
else
    echo -e "${YELLOW}⚠ Backend venv not found (run: cd backend && python3 -m venv venv)${NC}"
fi

if [ -d "agents/hound/.venv" ] || [ -d "hound/.venv" ]; then
    echo -e "${GREEN}✓ Hound venv exists${NC}"
else
    echo -e "${YELLOW}⚠ Hound venv not found (optional, Hound can use system Python)${NC}"
fi

if [ -d "agents/brama/venv" ]; then
    echo -e "${GREEN}✓ BRAMA venv exists${NC}"
else
    echo -e "${YELLOW}⚠ BRAMA venv not found (optional, BRAMA can use system Python)${NC}"
fi

echo ""

# Test 3: Check Python dependencies
echo "3. Checking Python dependencies..."

if [ -d "backend/venv" ]; then
    cd backend
    source venv/bin/activate 2>/dev/null
    if python3 -c "import fastapi" 2>/dev/null; then
        echo -e "${GREEN}✓ Backend dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠ Backend dependencies not installed (run: pip install -r requirements.txt)${NC}"
    fi
    deactivate 2>/dev/null
    cd ..
fi

echo ""

# Test 4: Check Hound CLI
echo "4. Testing Hound CLI..."
if [ -f "agents/hound/scripts/hound" ] || [ -f "hound/scripts/hound" ]; then
    HOUND_SCRIPT=$(find . -name "hound" -path "*/scripts/hound" 2>/dev/null | head -1)
    if [ -n "$HOUND_SCRIPT" ]; then
        echo -e "${GREEN}✓ Hound script found: $HOUND_SCRIPT${NC}"
        # Try to run hound project list (non-destructive)
        if python3 "$HOUND_SCRIPT" project list >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Hound CLI is functional${NC}"
        else
            echo -e "${YELLOW}⚠ Hound CLI found but may need configuration${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Hound script not found${NC}"
fi

echo ""

# Test 5: Check BRAMA wrapper
echo "5. Testing BRAMA wrapper..."
if [ -f "agents/brama/scan_url.py" ]; then
    echo -e "${GREEN}✓ BRAMA wrapper script found${NC}"
    if python3 -c "import sys; sys.path.insert(0, 'agents/brama'); import agentBrama" 2>/dev/null; then
        echo -e "${GREEN}✓ BRAMA imports successfully${NC}"
    else
        echo -e "${YELLOW}⚠ BRAMA may have dependency issues (check requirements.txt)${NC}"
    fi
else
    echo -e "${RED}✗ BRAMA wrapper script not found${NC}"
fi

echo ""

# Test 6: Check backend integration
echo "6. Testing backend integration..."
cd backend
if python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from main import HOUND_PATH, BRAMA_PATH
print('HOUND_PATH:', HOUND_PATH)
print('BRAMA_PATH:', BRAMA_PATH)
print('Hound exists:', HOUND_PATH.exists())
print('BRAMA exists:', BRAMA_PATH.exists())
" 2>/dev/null; then
    echo -e "${GREEN}✓ Backend can resolve agent paths${NC}"
else
    echo -e "${YELLOW}⚠ Backend path resolution needs checking${NC}"
fi
cd ..

echo ""

# Test 7: Check API keys (informational)
echo "7. Checking API keys (informational)..."
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "${GREEN}✓ ANTHROPIC_API_KEY is set${NC}"
else
    echo -e "${YELLOW}⚠ ANTHROPIC_API_KEY not set (required for BRAMA)${NC}"
fi

if [ -n "$VT_API_KEY" ]; then
    echo -e "${GREEN}✓ VT_API_KEY is set${NC}"
else
    echo -e "${YELLOW}⚠ VT_API_KEY not set (required for BRAMA)${NC}"
fi

if [ -n "$BRAVE_API_KEY" ]; then
    echo -e "${GREEN}✓ BRAVE_API_KEY is set${NC}"
else
    echo -e "${YELLOW}⚠ BRAVE_API_KEY not set (required for BRAMA)${NC}"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "${GREEN}✓ OPENAI_API_KEY is set${NC}"
else
    echo -e "${YELLOW}⚠ OPENAI_API_KEY not set (optional, for Hound)${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Integration Test Complete!"
echo ""
echo "Next steps:"
echo "1. Set required API keys (see docs/API_CLI_INTEGRATION.md)"
echo "2. Start backend: cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo "3. Start frontend: cd frontend && npm run dev"
echo "4. Test via browser: http://localhost:3000"
echo ""

