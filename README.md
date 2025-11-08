# Heimdall - Cybersecurity Co-Pilot & Vulnerability Scanner

Heimdall is an idiot-proof cybersecurity co-pilot and vulnerability scanner for small businesses, startups, and solo devs.

## Core Features

- **Scan Website** (External/Red-Team Mode): Surface scanning, endpoint fingerprinting, and web risk assessment
- **Audit Codebase** (Internal/Deep Mode): Deep codebase analysis with knowledge graphs and evidence tracking

## Architecture

- **Frontend**: Next.js with minimalist UI
- **Backend**: FastAPI orchestrating Hound and BRAMA agents
- **Hound**: Deep codebase analysis engine (https://github.com/scabench-org/hound)
- **BRAMA**: External threat analysis scanner (https://github.com/oborys/security-ai-agent-brama)

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git

### Automated Setup

Run the setup script:
```bash
./setup.sh
```

This will:
- Clone Hound and BRAMA repositories
- Set up Python virtual environment
- Install all dependencies

### Manual Setup

1. Clone the agent repositories:
```bash
mkdir -p agents
git clone https://github.com/scabench-org/hound.git agents/hound
git clone https://github.com/oborys/security-ai-agent-brama.git agents/brama
```

2. Setup backend:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Setup frontend:
```bash
cd frontend
npm install
cp .env.local.example .env.local  # Edit API URL if needed
```

4. Start services:

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Or use the convenience scripts:
```bash
./scripts/start-backend.sh
./scripts/start-frontend.sh
```

## Quick Start

### Option 1: Automated Start (Easiest)

```bash
# 1. Set your XAI API key
cd backend
cp .env.example .env
# Edit .env and add: XAI_API_KEY=your_key_here

# 2. Run the start script
cd ..
./START.sh
```

This starts both backend and frontend automatically!

### Option 2: Manual Start

```bash
# 1. Configure API key
cd backend
cp .env.example .env
# Edit .env and add: XAI_API_KEY=your_key_here

# 2. Start backend (Terminal 1)
source venv/bin/activate
uvicorn main:app --reload

# 3. Start frontend (Terminal 2)
cd ../frontend
npm install  # First time only
npm run dev
```

### 3. Open the Web App

Open **http://localhost:3000** in your browser.

You'll see two options:
- **Scan Website** - Uses BRAMA (red-team scanning + domain analysis)
- **Audit Codebase** - Uses Hound (deep codebase analysis)

## Configuration

**Only ONE API key required**: `XAI_API_KEY`

See [CONFIGURATION.md](CONFIGURATION.md) for details, or just:
1. Copy `backend/.env.example` to `backend/.env`
2. Add your XAI_API_KEY
3. Done!

## Documentation

- [Configuration Guide](CONFIGURATION.md) - API keys and setup
- [Integration Guide](docs/API_CLI_INTEGRATION.md) - Technical integration details
- [Red-Team Features](RED_TEAM_FEATURES.md) - BRAMA red-teaming capabilities

