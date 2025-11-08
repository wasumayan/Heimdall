# Heimdall Testing Guide

## Quick Test

### 1. Test BRAMA Red-Team Scanner Directly

```bash
# Set your xAI API key
export XAI_API_KEY="your_key_here"

# Test BRAMA scanner
cd agents/brama
python3 scan_url.py https://example.com
```

### 2. Test via Heimdall Backend

```bash
# Terminal 1: Start backend
cd backend
source venv/bin/activate
uvicorn main:app --reload

# Terminal 2: Test API
curl -X POST http://localhost:8000/scan-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

### 3. Test via Frontend UI

```bash
# Terminal 1: Backend (already running)
# Terminal 2: Start frontend
cd frontend
npm run dev
# Or use: PORT=3001 npm run dev if port 3000 is in use

# Open browser: http://localhost:3001 (or http://localhost:3000)
# Click "Scan Website" and enter a URL
```

## What Gets Scanned

When you scan a website, BRAMA now performs:

1. **Domain Threat Intelligence** (via xAI)
   - Malware/phishing analysis
   - Domain reputation check

2. **Security Headers Analysis**
   - Missing security headers
   - Header misconfigurations
   - Information disclosure

3. **SSL/TLS Certificate Check**
   - Certificate expiration
   - TLS version validation
   - Weak cipher detection

4. **Endpoint Discovery**
   - Common admin panels
   - API endpoints
   - Sensitive files
   - Backup directories

5. **Technology Stack Fingerprinting**
   - CMS detection (WordPress, Drupal, etc.)
   - Framework identification
   - Server software detection
   - Version information

6. **CORS Misconfiguration**
   - Wildcard origin issues
   - Credentials with wildcard
   - Origin reflection

7. **HTTP Methods Testing**
   - Dangerous methods (PUT, DELETE)
   - TRACE method detection
   - Method enumeration

8. **Information Disclosure**
   - Error messages
   - Debug information
   - Sensitive comments

## Expected Results

A successful scan will return:

```json
{
  "findings": [
    {
      "id": "brama_001",
      "severity": "high|medium|low",
      "title": "Security Finding Title",
      "description": "Detailed description...",
      "location": "https://example.com",
      "recommendation": "Actionable recommendation",
      "fix_suggestion": "Specific fix guidance"
    }
  ],
  "summary": {
    "total_findings": 5,
    "high": 1,
    "medium": 2,
    "low": 2
  }
}
```

## Troubleshooting

### BRAMA Import Errors
- Ensure you're in the `agents/brama` directory
- Check that `red_team_scanner.py` exists
- Verify Python dependencies: `pip install requests`

### No Findings Returned
- Check that the URL is accessible
- Verify XAI_API_KEY is set (for domain analysis)
- Check backend logs for errors

### Timeout Errors
- Some scans may take 30-60 seconds
- Increase timeout in `red_team_scanner.py` if needed
- Test with a simple URL first (e.g., example.com)

## Test URLs

Safe test URLs:
- `https://example.com` - Simple test site
- `https://httpbin.org` - HTTP testing service
- `https://jsonplaceholder.typicode.com` - API test site

**Note**: Only test on websites you own or have permission to test!

