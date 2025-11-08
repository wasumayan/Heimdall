# API Keys Summary - What You Actually Need

## ✅ Confirmed: Only ONE API Key Required!

**XAI_API_KEY** is the **ONLY** required API key for Heimdall to work.

## ❌ No Chrome/Chromium APIs Needed

- ✅ **No Chrome browser** - Heimdall uses pure Python libraries
- ✅ **No Chromium** - All scanning uses HTTP requests (requests library)
- ✅ **No browser automation** - No Selenium, Playwright, or Puppeteer
- ✅ **No Chrome DevTools API** - Not needed

**Note**: The `chromadb` package in requirements.txt is a **vector database** (for embeddings), not the Chrome browser. It's only used in educational mode, which isn't part of the main scanning flow.

## API Keys Breakdown

### Required (1 key)
- **XAI_API_KEY** - For AI analysis (Hound + BRAMA)

### Optional (0 keys needed)
All of these are **completely optional** and have fallbacks:

- **VT_API_KEY** - VirusTotal (optional, xAI fallback)
- **BRAVE_API_KEY** - Web search (optional, xAI fallback)
- **VOYAGE_API_KEY** - Only for educational mode (not used in scanning)
- **UMBRELLA_API_CLIENT/SECRET** - Cisco Umbrella (optional)
- **URL_HAUSE_KEY** - URLHaus (optional)

## What Works With Just XAI_API_KEY

✅ **Website Scanning (BRAMA)**:
- Domain threat intelligence (via xAI)
- Security headers analysis (no API needed)
- SSL/TLS certificate checks (no API needed)
- Endpoint discovery (no API needed)
- Technology stack fingerprinting (no API needed)
- CORS misconfiguration detection (no API needed)
- HTTP methods testing (no API needed)
- Information disclosure checks (no API needed)

✅ **Codebase Auditing (Hound)**:
- Full codebase analysis (via xAI)
- Knowledge graph building
- Vulnerability detection
- All Hound features

## Red-Team Scanner

The red-team scanner (`red_team_scanner.py`) uses **ZERO API keys**:
- Uses standard Python libraries: `requests`, `ssl`, `socket`
- No external API calls
- Works completely offline (except for HTTP requests to target site)

## Quick Start

```bash
# That's it! Just one key:
export XAI_API_KEY="your_key_here"

# Everything else works without additional keys
```

## Verification

To verify no Chrome/other APIs are needed:

```bash
# Check for browser automation
grep -r "selenium\|playwright\|puppeteer\|webdriver" agents/brama/
# Result: No matches

# Check red-team scanner dependencies
grep -E "import|from" agents/brama/red_team_scanner.py
# Result: Only standard libraries (requests, ssl, socket, re, json, urllib)
```

## Summary

- ✅ **1 API key required**: XAI_API_KEY
- ✅ **0 browser APIs**: No Chrome/Chromium needed
- ✅ **0 other required APIs**: Everything else is optional
- ✅ **Red-team scanner**: Works with zero API keys
- ✅ **Pure Python**: All scanning uses standard libraries

Heimdall is designed to work with minimal dependencies!

