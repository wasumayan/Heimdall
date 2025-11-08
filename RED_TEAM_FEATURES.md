# BRAMA Red-Team Features

BRAMA has been enhanced with comprehensive red-teaming capabilities for website security scanning.

## Red-Team Scanning Capabilities

### 1. Security Headers Analysis
- **X-Frame-Options**: Detects clickjacking vulnerabilities
- **X-Content-Type-Options**: Checks for MIME sniffing protection
- **Strict-Transport-Security (HSTS)**: Validates HTTPS enforcement
- **Content-Security-Policy (CSP)**: Checks XSS protection
- **Referrer-Policy**: Validates referrer information handling
- **Server Header Disclosure**: Detects information leakage
- **X-Powered-By Disclosure**: Identifies technology stack exposure

### 2. SSL/TLS Certificate Analysis
- Certificate expiration checks
- TLS version validation (flags weak versions)
- Cipher suite analysis (detects weak ciphers)
- Certificate chain validation

### 3. Endpoint Discovery
- Scans 20+ common paths:
  - Admin panels (`/admin`, `/wp-admin`, `/administrator`)
  - API endpoints (`/api`, `/graphql`, `/swagger`)
  - Sensitive files (`.git`, `.env`, `config.php`)
  - Backup files (`/backup`)
  - Documentation (`/docs`, `/swagger.json`)
- Identifies accessible endpoints
- Flags sensitive file exposure

### 4. Technology Stack Fingerprinting
- Detects 15+ technologies:
  - CMS: WordPress, Drupal, Joomla
  - Frameworks: React, Vue.js, Angular, Django, Flask, Laravel, Express
  - Servers: Apache, Nginx
  - Languages: PHP, Node.js, ASP.NET
- Version information detection
- Technology disclosure warnings

### 5. CORS Misconfiguration Detection
- Wildcard origin detection
- Credentials with wildcard (critical)
- Origin reflection vulnerabilities
- CSRF vulnerability identification

### 6. HTTP Methods Testing
- Tests: GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD, TRACE
- Flags dangerous methods (PUT, DELETE)
- Detects TRACE method (XST vulnerability)
- Identifies permissive method configurations

### 7. Information Disclosure Checks
- Error message detection (stack traces, fatal errors)
- Debug information disclosure
- Sensitive information in comments
- Version information leakage

## Usage

The red-team scanner is automatically integrated into BRAMA's website scanning:

```bash
# Via Heimdall API
curl -X POST http://localhost:8000/scan-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Direct BRAMA test
cd agents/brama
python3 scan_url.py https://example.com
```

## Output Format

All findings are returned in Heimdall's standard format:

```json
{
  "findings": [
    {
      "id": "brama_001",
      "severity": "high|medium|low",
      "title": "Finding Title",
      "description": "Detailed description",
      "location": "URL or endpoint",
      "recommendation": "Actionable recommendation",
      "fix_suggestion": "Specific fix guidance"
    }
  ],
  "summary": {
    "total_findings": 0,
    "high": 0,
    "medium": 0,
    "low": 0
  }
}
```

## Severity Levels

- **High**: Critical security issues (CORS misconfig, weak TLS, sensitive files exposed)
- **Medium**: Important security concerns (missing headers, endpoint discovery, dangerous methods)
- **Low**: Information disclosure (version info, technology stack)

## Integration

The red-team scanner is seamlessly integrated:
1. Domain threat intelligence (via xAI)
2. Comprehensive red-team security scan
3. Combined results in unified format
4. Automatic severity mapping
5. Actionable recommendations

## Future Enhancements

Potential additions:
- Subdomain enumeration
- Port scanning
- Authentication bypass testing
- SQL injection detection
- XSS vulnerability testing
- Advanced crawling
- Session management testing

