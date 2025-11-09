## Surface / Frontend Secret Exposure Agent

This agent inspects a running web application from inside a real browser context to surface credentials or sensitive data that leak to the frontend surface. It mirrors the structure of the network and injection agents while focusing on secret exposure inside local storage, session storage, cookies, the DOM, console logs, and network traffic.

### Capabilities
- Launches Chromium via Playwright (headless by default) and records the environment before and after page boot (new globals, current route, referrer meta tags).
- Enumerates `localStorage` and `sessionStorage` keys/values, redacts findings, and flags suspicious entries (`token`, `secret`, JWTs, `sk_live`, AWS keys, etc.) according to severity heuristics.
- Audits cookies via Playwright context APIs, flagging any cookie accessible to JavaScript (missing HttpOnly) and capturing other flag states (Secure, SameSite).
- Hooks into console logs, network requests, and responses to surface bearer tokens, API keys, or other high-risk patterns observed in headers, bodies, or `console.log` output.
- Detects sensitive tokens embedded within URLs or fragments and highlights permissive or missing `Referrer-Policy` configurations that could leak those values.
- Aggregates evidence with redacted previews, confidence scores, severities, and remediation text aligned with Heimdall's reporting schema.

### Running Locally
```bash
python agents/surface/surface_agent.py https://example.app --post-wait 3 --json
```
- Pass `--headful` to observe the browser session interactively.
- `--post-wait` adjusts the initial wait (seconds) after navigation before analysis begins; defaults to `2`.
- `--wait-until` mirrors Playwright's `page.goto(..., wait_until=...)` options (`load`, `domcontentloaded`, `networkidle`, `commit`).
- `--json` emits machine-readable output shaped like the other agents (`target`, `resolved_url`, `findings`, `summary`, etc.).

### Integrating with the Backend
```python
from agents.surface import SurfaceAgent, SurfaceAgentConfig

agent = SurfaceAgent(SurfaceAgentConfig(post_navigation_wait=1.5))
result = agent.scan("https://app.internal.example")

payload = result.to_dict()
for finding in payload["findings"]:
    print(finding["severity"], finding["kind"], finding["evidence"]["location"])
```
- Frontend exposure findings share the same severity enum as the network and injection agents for consistent rollups.
- `SurfaceScanResult.summary` includes severity counts and new global variables to aid UI rollups.
- Because Playwright is optional, install it only when the orchestrator needs browser context: `pip install playwright` and `playwright install chromium`.

### Checklist Covered
- ✅ Web storage enumeration (local/session storage)
- ✅ Cookie audit (HttpOnly, Secure, SameSite)
- ✅ Global scope/inline script inspection (new globals diffing)
- ✅ URL parameter/referrer checks
- ✅ Indexed cache placeholders (structure ready for extension)
- ✅ Console/network observation with secret detection
- ✅ Evidence aggregation with redacted previews and remediation text
- ✅ Structured JSON output compatible with Heimdall orchestrators


