## Heimdall Agents Overview

Heimdall ships with multiple specialized security agents. Each module targets a
different layer of the stack and can be run independently via CLI utilities or
integrated into the orchestration backend.

### Agent Catalog

- **Surface Agent (`agents.surface`)**
  - Playwright-driven browser automation that inspects rendered pages for UI/DOM
    issues and security misconfigurations.
  - Extracts links, scripts, and metadata to build a client-side view of the
    application.
  - CLI example:
    ```bash
    python agents/surface/surface_agent.py https://example.com --json
    ```

- **Network Agent (`agents.network`)**
  - Performs HTTP probing to examine headers, redirects, and transport security
    posture.
  - Flags missing security headers, permissive CORS, and insecure protocols.
  - CLI example:
    ```bash
    python agents/network/network_agent.py https://example.com --path /api/status --json
    ```

- **Auth Endpoint Agent (Agent C) (`agents.auth_endpoint`)**
  - Analyzes DOM and network logs to enumerate API endpoints and probe them
    without credentials.
  - Detects access-control gaps and sensitive data exposure for unauthenticated
    users.
  - CLI example:
    ```bash
    python agents/auth_endpoint/auth_endpoint_agent.py https://example.com \
      --rendered-dom fixtures/dom.html \
      --network-log fixtures/network_log.json \
      --json
    ```

- **Injection Agent (`agents.injection`)**
  - Focuses on reflected/stored/DOM XSS by combining HTTP payload fuzzing with
    browser evaluation.
  - Generates findings with context, payloads, and remediation suggestions.
  - CLI example:
    ```bash
    python agents/injection/injection_agent.py https://example.com --paths /search --json
    ```

- **BRAMA Agent (`agents.brama`)**
  - Legacy reconnaissance and scanning utilities (e-mail phishing detection,
    DMARC checks, etc.).
  - Provides higher-level workflows that leverage Brave search and other OSINT
    sources.
  - CLI usage varies per script; refer to the module-level README.

### Common Patterns

- All agents expose dataclass-based results with `.to_dict()` helpers for easy
  serialization into backend APIs.
- Optional LLM integrations are guarded behind environment variables (e.g.
  `OPENAI_API_KEY`, `XAI_API_KEY`).
- Most CLIs accept `--json` to emit machine-readable output compatible with the
  Heimdall backend ingestion pipeline.

