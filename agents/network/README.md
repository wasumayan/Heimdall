## Network / Request-Response Layer Agent

This agent inspects HTTP responses to surface risks that can be identified purely through network inspection. It purposefully lives outside of the BRAMA codebase so we can evolve it independently from the browser-surface work.

### Capabilities
- Probes a target URL (and optional extra paths) with `GET`, `HEAD`, and `OPTIONS`.
- Normalizes targets (adds `https://` when the scheme is omitted) and records redirects/latency.
- Flags missing security headers: `Strict-Transport-Security`, `Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`.
- Detects permissive CORS policies (wildcard origins, wildcard headers/methods, credentials without trusted origins).
- Reports insecure transport usage when responses are served over HTTP.
- Emits structured findings with severity levels that align with the backend schema (`id`, `severity`, `title`, `description`, `recommendation`, `evidence`).

### Running Locally
```bash
python agents/network/network_agent.py https://example.com --path /api/status --json
```
- `--path` can be repeated to probe additional endpoints relative to the base URL.
- `--timeout` adjusts the HTTP client timeout (seconds).
- `--insecure` disables TLS verification (useful for staging/self-signed targets).
- `--json` switches to machine-readable output that the backend can ingest directly.

### Integrating with the Backend
```python
from agents.network import AgentConfig, NetworkPolicyAgent

agent = NetworkPolicyAgent(AgentConfig(timeout=5))
result = agent.scan("https://target.example", extra_paths=["/healthz", "/api"])
payload = result.to_dict()  # share with FastAPI responses
```
- The `NetworkScanResult.summary` field already contains severity counts for UI rollups.
- Findings follow the same structure as existing Hound/BRAMA outputs, so the backend can merge them into a single report without additional transforms.
- Because the agent uses `httpx` synchronously, spawning it via `run_in_executor` (or a subprocess, similar to BRAMA) keeps the FastAPI event loop responsive.

## Auth / Endpoint Agent (Agent C)

This companion module discovers application/API endpoints from DOM artifacts and network logs, then probes them without credentials to uncover access-control gaps.

### Capabilities
- Parses Playwright-style `network_log` JSON and rendered DOM/JS (`fetch`, `axios`, `XMLHttpRequest`, form actions) to build an endpoint inventory with provenance tags.
- Applies heuristics for sensitive or premium paths (`/admin`, `/export`, `/premium`) and flags endpoints that originally required auth headers/cookies.
- Probes up to 30 endpoints per target with 3 concurrent workers (safe `GET/HEAD/OPTIONS` only), stripping cookies/auth headers automatically.
- Scores responses for PII/tokens/sensitive keywords and optionally calls the Grok/Claude classifier (via `XAI_API_KEY`) when regex hits are inconclusive.
- Emits findings when sensitive data is returned anonymously or when endpoints previously accessed with credentials now respond to unauthenticated requests. Each probe produces evidence with `sensitivity_tags`, `risk_level`, and remediation text.

### Usage
```python
from agents.network import AgentCConfig, AuthEndpointAgent

config = AgentCConfig(
    rendered_dom_path="fixtures/dom.html",
    network_log_path="fixtures/network_log.json",
    safe_endpoints_whitelist=("api.example.com",),
    concurrency=3,
)
agent = AuthEndpointAgent(config)
result = agent.scan("https://example.com")

for finding in result.findings:
    print(finding.severity, finding.title, finding.evidence["url"])
```
- Provide DOM/log artifacts directly (`rendered_dom`, `network_log`) or via the config paths shown above.
- `safe_endpoints_whitelist` restricts probing to trusted hosts (e.g., first-party API subdomains). Set `allow_external_hosts=True` in the config to override.
- When `XAI_API_KEY` is available, response snippets that fall below the regex confidence threshold are sent to the Grok classifier for tags (`email`, `jwt`, etc.). You can override the model via `XAI_CLASSIFIER_MODEL`.
- `result.summary_text`/`result.summary` mirror the structure of the network agent, and `result.evidence` contains the JSON report expected by the orchestrator.

### Playwright Wrapper

If you prefer to have the agent collect its own DOM + network data, the repository also includes a Playwright-powered wrapper:

```python
from agents.network import BrowserCaptureConfig, PlaywrightAuthEndpointAgent

capture_cfg = BrowserCaptureConfig(
    headless=True,
    extra_paths=("/pricing",),
    include_console=True,
)
agent = PlaywrightAuthEndpointAgent(capture_cfg)
result = agent.scan("https://example.com")
```

- Under the hood it launches Chromium via Playwright, logs all XHR/fetch traffic, grabs `page.content()`, and then feeds those artifacts to `AuthEndpointAgent`.
- The CLI (`python agents/network/playwright_auth_agent.py <url>`) accepts `--extra-path` hops, `--headful` browsing, and can emit JSON identical to the lower-level agent.
