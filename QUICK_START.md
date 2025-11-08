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

Frontend will run on: `http://localhost:3000`

## 🎯 Using the Web App

1. Open **http://localhost:3000** in your browser
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

## ✅ What's Included

### Frontend (Complete ✅)
- ✅ Beautiful, minimalist UI
- ✅ Two main entrypoints (Scan Website, Audit Codebase)
- ✅ Real-time results display
- ✅ Findings cards with severity colors
- ✅ Plain-language explanations
- ✅ Report download (HTML)
- ✅ Loading states and error handling
- ✅ Responsive design

### Backend (Complete ✅)
- ✅ FastAPI server
- ✅ BRAMA integration (with red-team features)
- ✅ Hound integration
- ✅ Subprocess-based agent calls
- ✅ Error handling and fallbacks
- ✅ Report generation

### Agents (Integrated ✅)
- ✅ BRAMA: Website scanning + red-teaming
- ✅ Hound: Codebase auditing

## 🧪 Test It

### Test Website Scan
1. Go to http://localhost:3000
2. Click "Scan Website"
3. Enter: `https://example.com`
4. Click "Scan Now"
5. View results!

### Test Codebase Audit
1. Go to http://localhost:3000
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

Everything is built and ready to use. Just:
1. Add your XAI_API_KEY to `backend/.env`
2. Start backend
3. Start frontend
4. Open browser and scan!

