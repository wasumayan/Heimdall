# Heimdall Backend

FastAPI orchestration layer for Hound and BRAMA security agents.

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

3. Run the server:
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
  "scan_options": {}
}
```

### `POST /audit-codebase-upload`
Upload and audit a codebase archive
- Form data with `file` field (zip/tar archive)

### `GET /result/{scan_id}`
Get scan/audit result by ID

### `GET /result/{scan_id}/report?format=html|pdf`
Get formatted report

## Development

The backend uses mock data by default. To integrate real agents:

1. Clone Hound and BRAMA repositories (see main README)
2. Update `run_hound_scan()` and `run_brama_scan()` functions in `main.py`
3. Test with real repositories and URLs

## Environment Variables

Create a `.env` file:
```
API_HOST=0.0.0.0
API_PORT=8000
HOUND_PATH=../agents/hound
BRAMA_PATH=../agents/brama
RESULTS_DIR=./results
```

