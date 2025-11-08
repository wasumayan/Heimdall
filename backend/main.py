"""
Heimdall Backend - FastAPI orchestration layer for Hound and BRAMA agents
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import subprocess
import os
import json
import tempfile
import shutil
from pathlib import Path
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Heimdall API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to agent repositories
# Try agents/hound first, fallback to root hound directory
_hound_path_agents = Path(__file__).parent.parent / "agents" / "hound"
_hound_path_root = Path(__file__).parent.parent / "hound"
HOUND_PATH = _hound_path_agents if _hound_path_agents.exists() else _hound_path_root

BRAMA_PATH = Path(__file__).parent.parent / "agents" / "brama"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class ScanWebsiteRequest(BaseModel):
    url: HttpUrl
    options: Optional[Dict[str, Any]] = {}


class AuditCodebaseRequest(BaseModel):
    github_repo: Optional[str] = None
    scan_options: Optional[Dict[str, Any]] = {}


class ScanResult(BaseModel):
    scan_id: str
    status: str
    findings: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str


@app.get("/")
async def root():
    return {
        "name": "Heimdall API",
        "version": "1.0.0",
        "endpoints": {
            "scan_website": "/scan-url",
            "audit_codebase": "/audit-codebase",
            "get_result": "/result/{scan_id}",
        },
    }


@app.post("/scan-url", response_model=ScanResult)
async def scan_website(request: ScanWebsiteRequest, background_tasks: BackgroundTasks):
    """
    Scan a website using BRAMA (external/red-team mode)
    """
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Run BRAMA scan
        result = await run_brama_scan(str(request.url), scan_id)
        
        return ScanResult(
            scan_id=scan_id,
            status="completed",
            findings=result.get("findings", []),
            summary=result.get("summary", {}),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")


@app.post("/audit-codebase")
async def audit_codebase(
    request: AuditCodebaseRequest,
    background_tasks: BackgroundTasks,
):
    """
    Audit a codebase using Hound (internal/deep mode)
    """
    scan_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        if request.github_repo:
            # Clone and scan GitHub repo
            result = await run_hound_github_scan(request.github_repo, scan_id)
        else:
            raise HTTPException(
                status_code=400, detail="GitHub repo URL or code upload required"
            )
        
        return ScanResult(
            scan_id=scan_id,
            status="completed",
            findings=result.get("findings", []),
            summary=result.get("summary", {}),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


@app.post("/audit-codebase-upload")
async def audit_codebase_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Audit uploaded codebase using Hound
    """
    scan_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Save uploaded file to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / file.filename
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Extract if needed (zip, tar, etc.)
            codebase_path = extract_codebase(temp_path, temp_dir)
            
            # Run Hound scan
            result = await run_hound_scan(codebase_path, scan_id)
        
        return ScanResult(
            scan_id=scan_id,
            status="completed",
            findings=result.get("findings", []),
            summary=result.get("summary", {}),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


@app.get("/result/{scan_id}")
async def get_result(scan_id: str):
    """
    Get scan/audit result by ID
    """
    result_file = RESULTS_DIR / f"{scan_id}.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    with open(result_file, "r") as f:
        return json.load(f)


@app.get("/result/{scan_id}/report")
async def get_report(scan_id: str, format: str = "html"):
    """
    Get formatted report (HTML or PDF)
    """
    result_file = RESULTS_DIR / f"{scan_id}.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    
    with open(result_file, "r") as f:
        result = json.load(f)
    
    if format == "html":
        report_path = generate_html_report(result, scan_id)
        return FileResponse(report_path, media_type="text/html")
    elif format == "pdf":
        report_path = generate_pdf_report(result, scan_id)
        return FileResponse(report_path, media_type="application/pdf")
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'html' or 'pdf'")


async def run_brama_scan(url: str, scan_id: str) -> Dict[str, Any]:
    """
    Run BRAMA external scan
    
    BRAMA is invoked via subprocess to avoid dependency conflicts.
    It runs in its own environment with its own dependencies.
    """
    # Check if BRAMA is available
    if not BRAMA_PATH.exists():
        # Return mock results for MVP
        return {
            "findings": [
                {
                    "id": "brama_001",
                    "severity": "high",
                    "title": "Potential XSS Vulnerability",
                    "description": "Found potential cross-site scripting vulnerability in form inputs",
                    "location": f"{url}/contact",
                    "recommendation": "Sanitize user inputs and implement Content Security Policy",
                    "fix_suggestion": "Use input validation and output encoding",
                },
                {
                    "id": "brama_002",
                    "severity": "medium",
                    "title": "Missing Security Headers",
                    "description": "Missing security headers (X-Frame-Options, CSP, etc.)",
                    "location": url,
                    "recommendation": "Add security headers to HTTP responses",
                    "fix_suggestion": "Configure web server to include security headers",
                },
            ],
            "summary": {
                "total_findings": 2,
                "high": 1,
                "medium": 1,
                "low": 0,
            },
        }
    
    try:
        # Run BRAMA via subprocess
        # Use the wrapper script that returns JSON
        brama_script = BRAMA_PATH / "scan_url.py"
        
        # Check if wrapper script exists, otherwise use the main agent
        if not brama_script.exists():
            # Fallback: try to use agentBrama.py directly (requires more setup)
            return {
                "findings": [],
                "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0},
                "error": "BRAMA scan_url.py wrapper not found. Please ensure scan_url.py exists in agents/brama/",
            }
        
        # Run BRAMA scan
        result = subprocess.run(
            [
                "python3",
                str(brama_script),
                str(url),
            ],
            cwd=str(BRAMA_PATH),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=os.environ.copy(),  # Pass through environment variables (API keys)
        )
        
        if result.returncode == 0:
            # Parse BRAMA's JSON output
            try:
                brama_data = json.loads(result.stdout)
                return transform_brama_output(brama_data, scan_id)
            except json.JSONDecodeError:
                # If JSON parsing fails, check stderr for errors
                error_msg = result.stderr or result.stdout
                return {
                    "findings": [],
                    "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0},
                    "error": f"Failed to parse BRAMA output: {error_msg[:200]}",
                }
        else:
            # BRAMA returned an error
            error_output = result.stderr or result.stdout
            return {
                "findings": [],
                "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0},
                "error": f"BRAMA scan failed: {error_output[:500]}",
            }
            
    except subprocess.TimeoutExpired:
        raise RuntimeError("BRAMA scan timed out")
    except FileNotFoundError:
        raise RuntimeError("BRAMA script not found. Ensure BRAMA is properly installed.")
    except Exception as e:
        # Return error but don't fail completely
        return {
            "findings": [],
            "summary": {"total_findings": 0, "high": 0, "medium": 0, "low": 0},
            "error": str(e),
        }


async def run_hound_scan(codebase_path: Path, scan_id: str) -> Dict[str, Any]:
    """
    Run Hound deep codebase audit
    
    Hound is invoked via subprocess to avoid dependency conflicts.
    It runs in its own environment with its own dependencies.
    """
    # Check if Hound is available
    if not HOUND_PATH.exists():
        # Return mock results for MVP
        return {
            "findings": [
                {
                    "id": "hound_001",
                    "severity": "critical",
                    "title": "SQL Injection Vulnerability",
                    "description": "Direct SQL query construction without parameterization",
                    "location": "app/database.py:42",
                    "code_snippet": "query = f'SELECT * FROM users WHERE id = {user_id}'",
                    "recommendation": "Use parameterized queries or ORM",
                    "fix_suggestion": "Replace with: query = 'SELECT * FROM users WHERE id = ?', params=[user_id]",
                },
                {
                    "id": "hound_002",
                    "severity": "high",
                    "title": "Hardcoded API Key",
                    "description": "API key found in source code",
                    "location": "config.py:15",
                    "code_snippet": "API_KEY = 'sk_live_1234567890'",
                    "recommendation": "Move secrets to environment variables",
                    "fix_suggestion": "Use: API_KEY = os.getenv('API_KEY')",
                },
            ],
            "summary": {
                "total_findings": 2,
                "critical": 1,
                "high": 1,
                "medium": 0,
                "low": 0,
            },
        }
    
    try:
        # Run Hound via subprocess to avoid dependency conflicts
        # Hound script handles its own environment
        hound_script = HOUND_PATH / "scripts" / "hound"
        
        # First, create a Hound project for this codebase
        project_name = f"heimdall_scan_{scan_id}"
        
        # Create project
        create_result = subprocess.run(
            [
                "python3",
                str(hound_script),
                "project",
                "create",
                project_name,
                str(codebase_path),
            ],
            cwd=str(HOUND_PATH),
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if create_result.returncode != 0:
            raise RuntimeError(f"Failed to create Hound project: {create_result.stderr}")
        
        # Run agent analysis
        agent_result = subprocess.run(
            [
                "python3",
                str(hound_script),
                "agent",
                "run",
                project_name,
            ],
            cwd=str(HOUND_PATH),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for analysis
        )
        
        if agent_result.returncode != 0:
            raise RuntimeError(f"Hound analysis failed: {agent_result.stderr}")
        
        # Read findings directly from Hound's project directory
        # Hound stores findings in ~/.hound/projects/{project_name}/hypotheses.json
        project_dir = Path.home() / ".hound" / "projects" / project_name
        hypotheses_file = project_dir / "hypotheses.json"
        
        if hypotheses_file.exists():
            try:
                with open(hypotheses_file, "r") as f:
                    hound_data = json.load(f)
                return transform_hound_output(hound_data, scan_id)
            except (json.JSONDecodeError, Exception) as e:
                # If reading fails, try to extract from agent output
                return parse_hound_text_output(agent_result.stdout, scan_id)
        else:
            # Fallback: try to extract findings from agent output
            return parse_hound_text_output(agent_result.stdout, scan_id)
            
    except subprocess.TimeoutExpired:
        raise RuntimeError("Hound scan timed out")
    except FileNotFoundError:
        raise RuntimeError("Hound script not found. Ensure Hound is properly installed.")
    except Exception as e:
        # Return error but don't fail completely
        return {
            "findings": [],
            "summary": {"total_findings": 0, "critical": 0, "high": 0, "medium": 0, "low": 0},
            "error": str(e),
        }


async def run_hound_github_scan(github_repo: str, scan_id: str) -> Dict[str, Any]:
    """
    Clone GitHub repo and run Hound scan
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "repo"
        
        # Clone repository
        subprocess.run(
            ["git", "clone", github_repo, str(repo_path)],
            check=True,
            capture_output=True,
        )
        
        # Run Hound scan
        return await run_hound_scan(repo_path, scan_id)


def extract_codebase(archive_path: Path, extract_dir: Path) -> Path:
    """
    Extract codebase archive (zip, tar, etc.)
    """
    # Simple implementation - assumes zip for now
    if archive_path.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir
    else:
        # Assume it's already a directory or single file
        return archive_path.parent


def generate_html_report(result: Dict[str, Any], scan_id: str) -> Path:
    """
    Generate HTML report from scan results
    """
    from jinja2 import Template
    
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heimdall Security Report - {{ scan_id }}</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }
            h1 { color: #333; }
            .summary { background: #f0f0f0; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .finding { border-left: 4px solid #ccc; padding: 15px; margin: 15px 0; background: #fafafa; }
            .finding.critical { border-color: #d32f2f; }
            .finding.high { border-color: #f57c00; }
            .finding.medium { border-color: #fbc02d; }
            .finding.low { border-color: #388e3c; }
            .severity { font-weight: bold; text-transform: uppercase; }
            .code { background: #f5f5f5; padding: 10px; font-family: monospace; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Heimdall Security Report</h1>
            <p><strong>Scan ID:</strong> {{ scan_id }}</p>
            <p><strong>Timestamp:</strong> {{ result.timestamp }}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Findings: {{ result.summary.total_findings }}</p>
                <p>Critical: {{ result.summary.get('critical', 0) }}</p>
                <p>High: {{ result.summary.high }}</p>
                <p>Medium: {{ result.summary.medium }}</p>
                <p>Low: {{ result.summary.low }}</p>
            </div>
            
            <h2>Findings</h2>
            {% for finding in result.findings %}
            <div class="finding {{ finding.severity }}">
                <h3>{{ finding.title }}</h3>
                <p class="severity">Severity: {{ finding.severity }}</p>
                <p>{{ finding.description }}</p>
                {% if finding.location %}
                <p><strong>Location:</strong> {{ finding.location }}</p>
                {% endif %}
                {% if finding.code_snippet %}
                <div class="code">{{ finding.code_snippet }}</div>
                {% endif %}
                <p><strong>Recommendation:</strong> {{ finding.recommendation }}</p>
                {% if finding.fix_suggestion %}
                <p><strong>Fix Suggestion:</strong> {{ finding.fix_suggestion }}</p>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    
    template = Template(template_str)
    html_content = template.render(result=result, scan_id=scan_id)
    
    report_path = RESULTS_DIR / f"{scan_id}_report.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    
    return report_path


def transform_hound_output(hound_data: Dict[str, Any], scan_id: str) -> Dict[str, Any]:
    """
    Transform Hound's hypotheses.json format to Heimdall's expected format.
    
    Hound stores findings in hypotheses.json with structure:
    {
        "hypotheses": [
            {
                "id": "...",
                "status": "confirmed" | "rejected" | "uncertain",
                "title": "...",
                "description": "...",
                "confidence": 0.0-1.0,
                "evidence": [...],
                "file": "...",
                "line": ...,
                ...
            }
        ]
    }
    """
    findings = []
    
    # Hound stores hypotheses in a list
    hypotheses = hound_data.get("hypotheses", [])
    
    # Only include confirmed vulnerabilities (status == "confirmed")
    for hyp in hypotheses:
        status = hyp.get("status", "").lower()
        
        # Only process confirmed findings
        if status == "confirmed":
            # Extract location information
            file_path = hyp.get("file", hyp.get("location", "unknown"))
            line_num = hyp.get("line", hyp.get("line_number", 0))
            location = f"{file_path}:{line_num}" if file_path != "unknown" else "unknown"
            
            # Extract code snippet from evidence if available
            code_snippet = ""
            evidence = hyp.get("evidence", [])
            if evidence and isinstance(evidence, list):
                # Try to find code evidence
                for ev in evidence:
                    if isinstance(ev, dict) and "code" in ev:
                        code_snippet = ev.get("code", "")
                        break
                    elif isinstance(ev, str) and ("def " in ev or "class " in ev or "=" in ev):
                        code_snippet = ev
                        break
            
            # Determine severity from confidence or explicit severity field
            confidence = hyp.get("confidence", 0.5)
            if confidence >= 0.9:
                severity = "critical"
            elif confidence >= 0.7:
                severity = "high"
            elif confidence >= 0.5:
                severity = "medium"
            else:
                severity = "low"
            
            # Override with explicit severity if present
            if "severity" in hyp:
                severity = map_severity(hyp["severity"])
            
            findings.append({
                "id": hyp.get("id", f"hound_{len(findings)}"),
                "severity": severity,
                "title": hyp.get("title", hyp.get("name", "Security Issue")),
                "description": hyp.get("description", hyp.get("summary", hyp.get("rationale", ""))),
                "location": location,
                "code_snippet": code_snippet or hyp.get("code", ""),
                "recommendation": hyp.get("recommendation", hyp.get("fix_guidance", "")),
                "fix_suggestion": hyp.get("fix", hyp.get("fix_suggestion", "")),
            })
    
    # Calculate summary
    summary = {
        "total_findings": len(findings),
        "critical": sum(1 for f in findings if f["severity"] == "critical"),
        "high": sum(1 for f in findings if f["severity"] == "high"),
        "medium": sum(1 for f in findings if f["severity"] == "medium"),
        "low": sum(1 for f in findings if f["severity"] == "low"),
    }
    
    return {
        "findings": findings,
        "summary": summary,
    }


def parse_hound_text_output(text_output: str, scan_id: str) -> Dict[str, Any]:
    """
    Fallback: Parse Hound's text output if JSON is not available.
    This is a basic parser - may need refinement based on actual output.
    """
    findings = []
    lines = text_output.split("\n")
    
    # Basic pattern matching for vulnerabilities in text output
    # Adjust based on Hound's actual text format
    current_finding = None
    for line in lines:
        if "CRITICAL" in line.upper() or "HIGH" in line.upper():
            if current_finding:
                findings.append(current_finding)
            severity = "critical" if "CRITICAL" in line.upper() else "high"
            current_finding = {
                "id": f"hound_{len(findings)}",
                "severity": severity,
                "title": line.strip(),
                "description": "",
                "location": "",
                "code_snippet": "",
                "recommendation": "",
                "fix_suggestion": "",
            }
        elif current_finding and line.strip():
            if not current_finding["description"]:
                current_finding["description"] = line.strip()
            elif ":" in line and not current_finding["location"]:
                current_finding["location"] = line.strip()
    
    if current_finding:
        findings.append(current_finding)
    
    summary = {
        "total_findings": len(findings),
        "critical": sum(1 for f in findings if f["severity"] == "critical"),
        "high": sum(1 for f in findings if f["severity"] == "high"),
        "medium": sum(1 for f in findings if f["severity"] == "medium"),
        "low": sum(1 for f in findings if f["severity"] == "low"),
    }
    
    return {
        "findings": findings,
        "summary": summary,
    }


def transform_brama_output(brama_data: Dict[str, Any], scan_id: str) -> Dict[str, Any]:
    """
    Transform BRAMA's output format to Heimdall's expected format.
    """
    # BRAMA wrapper already returns the correct format, but we'll ensure consistency
    findings = brama_data.get("findings", [])
    summary = brama_data.get("summary", {})
    
    # Ensure all findings have required fields
    for finding in findings:
        if "severity" in finding:
            finding["severity"] = map_severity(finding["severity"])
    
    return {
        "findings": findings,
        "summary": summary,
    }


def map_severity(severity: str) -> str:
    """
    Map Hound's severity levels to Heimdall's standard levels.
    """
    severity_lower = str(severity).lower()
    
    if severity_lower in ["critical", "crit", "p0", "0"]:
        return "critical"
    elif severity_lower in ["high", "h", "p1", "1"]:
        return "high"
    elif severity_lower in ["medium", "med", "m", "p2", "2"]:
        return "medium"
    elif severity_lower in ["low", "l", "p3", "3", "info", "informational"]:
        return "low"
    else:
        return "medium"  # Default


def generate_pdf_report(result: Dict[str, Any], scan_id: str) -> Path:
    """
    Generate PDF report from scan results
    """
    # For MVP, generate HTML first, then convert to PDF if needed
    # Or use a library like reportlab
    html_path = generate_html_report(result, scan_id)
    # TODO: Convert HTML to PDF using weasyprint or similar
    return html_path  # Placeholder


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

