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
