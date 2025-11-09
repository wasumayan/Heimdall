# Heimdall — Autonomous Security Co-Pilot

Modern devs ship fast — Heimdall keeps it safe. Built for vibecoders, lean security teams, and anyone shipping product at velocity, Heimdall pairs autonomous browser agents with a code-aware audit brain so you can harden production without slowing launches.

## What Heimdall Delivers

- **Runtime red-teaming**: Surface, Network, Endpoint, and Injection agents drive real browsers to uncover client-side, transport, auth, and injection bugs in live environments.
- **Deep code intelligence**: Hound ingests repositories, builds semantic knowledge graphs, and investigates hypotheses to trace vulnerabilities back to exact lines of code.
- **Evidence-first reporting**: Every finding carries reproducible evidence, remediation guidance, and full provenance so fixes are fast and defensible.
- **Live telemetry & steering**: Real-time SSE streams let you watch investigations unfold and steer agents mid-run when you spot an interesting lead.
- **Single-operator friendly**: Defaults, scripts, and guardrails mean a solo dev can run an enterprise-grade security review in minutes.

## Architecture At A Glance

- **Frontend (`frontend/`)**: Vite + React 18 + Tailwind + Motion. Blueprint-inspired UI that lets you launch scans, monitor telemetry, explore findings, and export reports.
- **Backend (`backend/`)**: FastAPI orchestration layer that wraps the Hound CLI, BRAMA web scanner, telemetry proxy endpoints, graph retrieval APIs, and report generation.
- **Agent Suite (`agents/`)**:
  - `surface/`: Browser automation that inspects rendered DOMs for runtime issues.
  - `network/`: HTTP header and transport analysis with CORS/cert heuristics.
  - `auth_endpoint/`: Endpoint mapper that correlates traffic with missing auth controls.
  - `injection/`: Playwright-backed fuzzing for XSS and contextual payloads.
  - `brama/`: External-facing red-team scanner for domains.
- **Hound (`hound/`)**: Embedded fork of the open-source Hound engine powering code graphing, planning, strategizing, and evidence-led audits.

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- xAI API key (`XAI_API_KEY`) — grab one at https://console.x.ai/

### Option 1 · Guided Launch (recommended)

```bash
# set up your API key (first run)
cd backend
cp .env.example .env
# edit backend/.env and add: XAI_API_KEY=your_key_here

cd ..
./START.sh
```

`START.sh` provisions a Python virtualenv, installs requirements, boots the FastAPI backend on `http://localhost:8000`, and then launches the Vite frontend on the first open port (3000/3001).

### Option 2 · Manual Control

Terminal 1 – backend:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # first run only
# add XAI_API_KEY to backend/.env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 – frontend:

```bash
cd frontend
npm install        # first run only
PORT=3001 npm run dev
```

Open `http://localhost:3001` (or whatever port Vite reports) to launch the console. Choose **Scan Website** for BRAMA-powered runtime recon or **Audit Codebase** for Hound-driven graph analysis.

## Feature Highlights

- **Autonomous investigations**: Hound breaks repositories into semantic fragments, plans multi-hop investigations, and confirms exploitable paths with reviewer models.
- **Graph-native visualizations**: Knowledge graphs and card stores are exposed through `/project/{name}/graph` endpoints for frontend visualization or custom pipelines.
- **Telemetry proxy**: `/telemetry/{project}/events` streams Hound’s SSE feed so the UI can render live progress and let you steer missions.
- **Report & export**: Findings are normalized, summarized, and exportable as HTML/PDF reports directly from the API.
- **Scriptable workflows**: Bash helpers in `scripts/` wrap common operations (start services, run full integrations, exercise BRAMA) for CI or local automation.

## Configuration & API Keys

- **Required**: `XAI_API_KEY` (set in `backend/.env`).
- **Optional add-ons**: `VT_API_KEY`, `BRAVE_API_KEY`, `VOYAGE_API_KEY`, Cisco Umbrella credentials, and more — see `CONFIGURATION.md` and `API_KEYS_SUMMARY.md` for the full matrix.

Heimdall only needs the xAI key out of the box; optional keys unlock richer enrichment for BRAMA and investigative agents.

## Testing & Operational Guides

- `QUICK_START.md` — condensed onboarding steps.
- `TESTING_GUIDE.md` — recommended flows for validating agents end-to-end.
- `RED_TEAM_FEATURES.md` — deep dive on BRAMA’s external attack surface coverage.
- `WHITELIST_GUIDE.md` — how the Hound whitelist builder works and how to override it.
- `docs/INTEGRATION.md` & `docs/API_CLI_INTEGRATION.md` — integrate Heimdall into CI/CD or invoke the API programmatically.

## Repository Map

```
backend/      FastAPI service, integrations, orchestration
frontend/     Vite + React UI for running scans and reviewing evidence
agents/       Runtime agent implementations (surface, network, auth-endpoint, injection, brama)
hound/        Embedded Hound engine with CLI, graph builder, strategist, telemetry
scripts/      Helper scripts for starting services and running automated tests
docs/         Additional configuration and integration guides
```

## Contributing

Issues and PRs are welcome. If you're extending agent capabilities or integrating new scanners, start by reviewing `CONTRIBUTING.md` and the relevant agent README in `agents/`.

---

Heimdall keeps the vibe high and the attack surface low — plug it into your stack, point it at prod or your repo, and let the autonomous agents get to work.

<sub>**Credits**: Heimdall uses [Hound](https://github.com/muellerberndt/hound) by [Bernhard Mueller](https://github.com/muellerberndt) for codebase analysis. See [paper](https://arxiv.org/html/2510.09633v1) and [walkthrough](https://muellerberndt.medium.com/hunting-for-security-bugs-in-code-with-ai-agents-a-full-walkthrough-a0dc24e1adf0). Licensed under Apache 2.0.</sub>
