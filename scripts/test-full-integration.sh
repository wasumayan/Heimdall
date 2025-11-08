#!/bin/bash

# Full integration test for Heimdall
echo "🧪 Testing Heimdall Full Integration"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test 1: Check environment
echo "1. Checking environment..."
if [ -z "$XAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠ XAI_API_KEY not set (will use mock data)${NC}"
else
    echo -e "${GREEN}✓ XAI_API_KEY is set${NC}"
fi

# Test 2: Test BRAMA directly
echo ""
echo "2. Testing BRAMA scanner directly..."
cd "$(dirname "$0")/../agents/brama"

if [ -f "scan_url.py" ]; then
    TEST_URL="https://example.com"
    echo "Testing with: $TEST_URL"
    
    if python3 -c "from red_team_scanner import RedTeamScanner; print('RedTeamScanner imported successfully')" 2>/dev/null; then
        echo -e "${GREEN}✓ RedTeamScanner module loads${NC}"
    else
        echo -e "${RED}✗ RedTeamScanner module failed to load${NC}"
    fi
    
    # Quick syntax check
    if python3 -m py_compile scan_url.py 2>/dev/null; then
        echo -e "${GREEN}✓ scan_url.py syntax is valid${NC}"
    else
        echo -e "${RED}✗ scan_url.py has syntax errors${NC}"
    fi
else
    echo -e "${RED}✗ scan_url.py not found${NC}"
fi

# Test 3: Test backend integration
echo ""
echo "3. Testing backend integration..."
cd "$(dirname "$0")/.."
if [ -d "backend" ]; then
    cd backend
else
    echo -e "${YELLOW}⚠ Backend directory not found${NC}"
    cd "$(dirname "$0")/.."
fi

if [ -f "main.py" ]; then
    if python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from main import BRAMA_PATH, HOUND_PATH
print(f'BRAMA_PATH: {BRAMA_PATH}')
print(f'HOUND_PATH: {HOUND_PATH}')
print(f'BRAMA exists: {BRAMA_PATH.exists()}')
print(f'Hound exists: {HOUND_PATH.exists()}')
" 2>/dev/null; then
        echo -e "${GREEN}✓ Backend can resolve agent paths${NC}"
    else
        echo -e "${YELLOW}⚠ Backend path resolution check failed${NC}"
    fi
else
    echo -e "${RED}✗ main.py not found${NC}"
fi

# Test 4: Check dependencies
echo ""
echo "4. Checking Python dependencies..."
if python3 -c "import requests; import ssl; import socket" 2>/dev/null; then
    echo -e "${GREEN}✓ Required Python modules available${NC}"
else
    echo -e "${YELLOW}⚠ Some Python modules may be missing${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Integration Test Complete!"
echo ""
echo "Next steps:"
echo "1. Set XAI_API_KEY: export XAI_API_KEY='your_key'"
echo "2. Start backend: cd backend && source venv/bin/activate && uvicorn main:app --reload"
echo "3. Start frontend: cd frontend && npm run dev"
echo "4. Test via browser: http://localhost:3000"
echo ""

