# Heimdall Quick Start Guide

## 🚀 Get Running in 3 Steps

### Step 1: Set Your API Key

```bash
cd backend
cp .env.example .env
# Edit .env and add your XAI_API_KEY
# That's it! Only one key needed.
```

Edit `backend/.env`:
```bash
XAI_API_KEY=your_actual_xai_key_here
```

### Step 2: Start Backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```

Backend will run on: `http://localhost:8000`

### Step 3: Start Frontend

In a **new terminal**:

```bash
cd frontend
npm install  # First time only
npm run dev
```

Frontend will run on: `http://localhost:3001` (or 3000 if available)

**Note**: If port 3000 is already in use, Next.js will automatically use 3001.

## 🎯 Using the Web App

1. Open **http://localhost:3001** (or **http://localhost:3000**) in your browser
2. You'll see two options:
   - **Scan Website** - Uses BRAMA (red-team scanning + domain analysis)
   - **Audit Codebase** - Uses Hound (deep codebase analysis)

### Scan Website
- Enter any URL (e.g., `https://example.com`)
- Click "Scan Now"
- Get comprehensive security findings:
  - Domain threat intelligence
  - Security headers analysis
  - SSL/TLS certificate checks
  - Endpoint discovery
  - Technology stack fingerprinting
  - CORS misconfiguration
  - HTTP methods testing
  - Information disclosure

### Audit Codebase
- Enter GitHub repository URL (e.g., `https://github.com/username/repo`)
- Click "Start Audit"
- Get deep codebase analysis:
  - Knowledge graph-based analysis
  - Vulnerability detection
  - Evidence tracking
  - Auto-fix suggestions

## ✅ What's Included (MVP Complete)

### Frontend (Complete ✅)
- ✅ Beautiful, minimalist UI built with Next.js/React
- ✅ Two main entrypoints (Scan Website, Audit Codebase)
- ✅ Real-time results display with live updates
- ✅ Findings cards with severity-based color coding
- ✅ Plain-language explanations (no technical jargon)
- ✅ Report download (HTML format)
- ✅ Loading states and error handling
- ✅ Fully responsive design
- ✅ Deployed and ready to use

### Backend (Complete ✅)
- ✅ FastAPI server with full API endpoints
- ✅ BRAMA integration (with comprehensive red-team features)
- ✅ Hound integration (deep codebase analysis)
- ✅ Subprocess-based agent calls (isolated environments)
- ✅ Error handling and graceful fallbacks
- ✅ Report generation (HTML format)
- ✅ Environment variable management (.env support)

### Agents (Fully Integrated ✅)
- ✅ BRAMA: Website scanning + red-teaming (7 scan types)
- ✅ Hound: Codebase auditing with knowledge graphs
- ✅ Both agents use xAI (Grok) for AI analysis
- ✅ Virtual environment isolation
- ✅ CLI wrapper scripts for subprocess calls

## 🧪 Test It

### Test Website Scan
1. Go to http://localhost:3001 (or http://localhost:3000)
2. Click "Scan Website"
3. Enter: `https://example.com`
4. Click "Scan Now"
5. View results!

### Test Codebase Audit
1. Go to http://localhost:3001 (or http://localhost:3000)
2. Click "Audit Codebase"
3. Enter: `https://github.com/username/repo` (any public repo)
4. Click "Start Audit"
5. View results!

## 📝 Notes

- **Only XAI_API_KEY required** - Everything else is optional
- **No Chrome/Chromium needed** - Pure Python libraries
- **Mock data fallback** - Works even if agents aren't fully configured
- **All results private** - Nothing is shared externally

## 🐛 Troubleshooting

### Backend won't start
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend won't start
```bash
cd frontend
npm install
npm run dev
```

### No results showing
- Check backend logs for errors
- Verify XAI_API_KEY is set: `echo $XAI_API_KEY`
- Check browser console for errors

### API connection error
- Ensure backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in frontend (defaults to http://localhost:8000)

## 🎉 You're Ready!

**MVP Status**: Everything is built, integrated, and ready to use!

Just:
1. Add your XAI_API_KEY to `backend/.env`
2. Run `./START.sh` (or start manually)
3. Open browser at http://localhost:3001
4. Start scanning!

**Repository**: https://github.com/wasumayan/Heimdall

