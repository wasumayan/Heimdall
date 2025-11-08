# API Requirements - Final Confirmation

## ✅ Confirmed: Only XAI_API_KEY Required

### Required API Keys: 1
- **XAI_API_KEY** - For AI analysis (Hound + BRAMA domain analysis)

### Optional API Keys: 0 (All have fallbacks)
- VT_API_KEY - Optional (xAI fallback)
- BRAVE_API_KEY - Optional (xAI fallback)  
- VOYAGE_API_KEY - Only for educational mode (not used in scanning)
- UMBRELLA_API_CLIENT/SECRET - Optional
- URL_HAUSE_KEY - Optional

## ✅ No Browser APIs Needed

- ❌ No Chrome browser
- ❌ No Chromium
- ❌ No Selenium
- ❌ No Playwright
- ❌ No Puppeteer
- ❌ No WebDriver

**Note**: `chromadb` in requirements.txt is a vector database library, not Chrome browser.

## Red-Team Scanner Dependencies

The red-team scanner uses only standard Python libraries:
- `requests` - HTTP requests
- `ssl` - SSL/TLS certificate checks
- `socket` - Network connections
- `re` - Regular expressions
- `json` - JSON parsing
- `urllib.parse` - URL parsing

**Zero API keys needed** for red-team scanning!

## Summary

✅ **1 API key**: XAI_API_KEY  
✅ **0 browser APIs**: Pure Python libraries only  
✅ **0 other required APIs**: Everything optional with fallbacks  

Heimdall works with minimal dependencies!
