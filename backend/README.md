# Heimdall Backend

FastAPI orchestration layer for Hound and BRAMA security agents.

## Current Status: Fully Integrated ✅

- ✅ Hound integration (codebase auditing via CLI)
- ✅ BRAMA integration (website scanning + red-teaming)
- ✅ **Whitelist Builder Integration**: Auto-generates file whitelists within LOC budget
- ✅ Subprocess-based agent calls
- ✅ Error handling and fallbacks
- ✅ Report generation
- ✅ **Telemetry Proxy**: SSE streaming endpoints for real-time audit events
- ✅ **Graph Data API**: Endpoints for knowledge graph retrieval

## Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add: XAI_API_KEY=your_key_here
```

4. Run the server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### `GET /`
Health check and API information

### `POST /scan-url`
Scan a website using BRAMA
```json
{
  "url": "https://example.com",
  "options": {}
}
```

### `POST /audit-codebase`
Audit a codebase using Hound
```json
{
  "github_repo": "https://github.com/username/repo",
  "auto_generate_whitelist": true,
  "whitelist_loc_budget": 50000,
  "graph_file_filter": null,
  "build_graphs": true,
  "iterations": 20,
  "audit_mode": "sweep",
  "telemetry": true,
  ...
}
```

**Whitelist Options**:
- `auto_generate_whitelist` (default: `true`): Auto-generate file whitelist within LOC budget
- `whitelist_loc_budget` (default: `50000`): Maximum lines of code to include
- `graph_file_filter` (optional): Manual comma-separated file list (overrides auto-generation)

### `POST /audit-codebase-upload`
Upload and audit a codebase archive
- Form data with `file` field (zip/tar archive)

### `GET /result/{scan_id}`
Get scan/audit result by ID

### `GET /result/{scan_id}/report?format=html|pdf`
Get formatted report

### `GET /project/{project_name}/graphs`
List available knowledge graphs for a project

### `GET /project/{project_name}/graph/{graph_name}`
Get specific graph data and card store

### `GET /telemetry/{project_name}/events`
Proxies SSE stream from Hound's telemetry server (real-time audit events)

### `GET /telemetry/{project_name}/status`
Check if telemetry server is running for a project

### `POST /telemetry/{project_name}/steer`
Forward steering commands to Hound telemetry server

## Agent Integration

Both agents are fully integrated:

### Hound Integration
- Calls Hound CLI via subprocess (`project create`, `graph build`, `agent audit`, `finalize`)
- **Whitelist Builder**: Auto-generates file whitelist if not provided (within LOC budget)
- Creates project and runs analysis
- Reads findings from `~/.hound/projects/{name}/hypotheses.json`
- Transforms output to Heimdall format (preserves all Hound fields)
- **Telemetry Support**: Proxies SSE streams from Hound's telemetry server
- **Graph Data**: Provides endpoints for knowledge graph retrieval

### BRAMA Integration
- Calls BRAMA wrapper script (`agents/brama/scan_url.py`)
- Performs domain analysis + red-team scanning
- Returns comprehensive security findings
- Transforms output to Heimdall format

## Environment Variables

Required:
- `XAI_API_KEY` - xAI API key for AI analysis (required)

Optional (for enhanced features):
- `VT_API_KEY` - VirusTotal API key
- `BRAVE_API_KEY` - Brave Search API key
- `VOYAGE_API_KEY` - Voyage AI (educational mode only)
- `UMBRELLA_API_CLIENT` / `UMBRELLA_API_SECRET` - Cisco Umbrella
- `URL_HAUSE_KEY` - URLHaus malware database

See `CONFIGURATION.md` in the root directory for details.

