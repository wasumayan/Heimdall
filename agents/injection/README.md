# Heimdall Injection Agent

This standalone agent probes a target for cross-site scripting (XSS) vectors by
combining HTTP inspection with Playwright-powered browser automation.

## Dependencies

Install dependencies and download the Chromium runtime required by Playwright:

```bash
pip install -r agents/injection/requirements.txt
playwright install
```

The agent also reuses `httpx` (already required by other agents).

## Usage

Run the agent against a target domain or URL:

```bash
python -m agents.injection.injection_agent https://example.com --json
```

Key flags:

- `--path /login` – add extra relative paths to probe.
- `--headful` – launch Playwright with a visible browser for debugging.
- `--insecure` – disable TLS verification (not recommended).
- `--timeout` – adjust HTTP and Playwright timeouts.
- `--json` – emit structured JSON output.

If Playwright is not installed, the agent will still perform HTTP/HTML analysis
but will skip dynamic DOM-based checks and warn via an informational finding.


