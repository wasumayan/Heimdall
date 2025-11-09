# Heimdall - Cybersecurity Co-Pilot & Vulnerability Scanner

Heimdall is an idiot-proof cybersecurity co-pilot and vulnerability scanner for small businesses, startups, and solo devs.

## Core Features

- **Scan Website** (External/Red-Team Mode): Surface scanning, endpoint fingerprinting, and web risk assessment
- **Audit Codebase** (Internal/Deep Mode): Deep codebase analysis with knowledge graphs and evidence tracking

## Architecture

- **Frontend**: Next.js with minimalist UI
- **Backend**: FastAPI orchestrating Hound and BRAMA agents
- **Hound**: Deep codebase analysis engine (https://github.com/muellerberndt/hound)
- **BRAMA**: External threat analysis scanner (https://github.com/oborys/security-ai-agent-brama)

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- xAI API Key (get from https://console.x.ai/)

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

**Note**: If port 3000 is in use, the script will use port 3001 automatically.

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
PORT=3001 npm run dev  # Use 3001 if 3000 is occupied
```

### 3. Open the Web App

Open **http://localhost:3001** (or **http://localhost:3000** if available) in your browser.

You'll see two options:
- **Scan Website** - Uses BRAMA (red-team scanning + domain analysis)
- **Audit Codebase** - Uses Hound (deep codebase analysis)

## Current Status: MVP Complete ✅

- ✅ **Frontend**: Complete with minimalist UI, real-time results, report export, whitelist configuration modal
- ✅ **Backend**: FastAPI server with full Hound and BRAMA integration, whitelist builder integration
- ✅ **Hound**: Deep codebase analysis with knowledge graphs, telemetry, all CLI options exposed
- ✅ **BRAMA**: Website scanning with comprehensive red-team features
- ✅ **Whitelist Builder**: Auto-generation with LOC budget, manual override option
- ✅ **Telemetry**: Real-time audit monitoring with live event streaming
- ✅ **Graph Visualization**: Interactive D3.js knowledge graph viewer
- ✅ **Documentation**: Complete setup and configuration guides
- ✅ **GitHub**: Repository live at https://github.com/wasumayan/Heimdall

## Configuration

**Only ONE API key required**: `XAI_API_KEY`

See [CONFIGURATION.md](CONFIGURATION.md) for details, or just:
1. Copy `backend/.env.example` to `backend/.env`
2. Add your XAI_API_KEY
3. Done!

## Documentation

- [Quick Start Guide](QUICK_START.md) - Get running in 3 steps
- [Configuration Guide](CONFIGURATION.md) - API keys and setup
- [Red-Team Features](RED_TEAM_FEATURES.md) - BRAMA red-teaming capabilities
- [Testing Guide](TESTING_GUIDE.md) - How to test the system
- [API Keys Summary](API_KEYS_SUMMARY.md) - What you actually need

---

<sub>**Credits**: Heimdall uses [Hound](https://github.com/muellerberndt/hound) by [Bernhard Mueller](https://github.com/muellerberndt) for codebase analysis. See [paper](https://arxiv.org/html/2510.09633v1) and [walkthrough](https://muellerberndt.medium.com/hunting-for-security-bugs-in-code-with-ai-agents-a-full-walkthrough-a0dc24e1adf0). Licensed under Apache 2.0.</sub>

