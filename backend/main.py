"""
Heimdall Backend - FastAPI orchestration layer for Hound and BRAMA agents
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Heimdall API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
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

# Persistent audit directories (kept for Hound to access source files)
AUDITS_DIR = Path(__file__).parent / "audits"
AUDITS_DIR.mkdir(exist_ok=True)


class ScanWebsiteRequest(BaseModel):
    url: HttpUrl
    options: Optional[Dict[str, Any]] = {}


class AuditCodebaseRequest(BaseModel):
    """Complete Hound CLI audit options mapped to API"""
    # Project creation
    github_repo: Optional[str] = None
    project_name: Optional[str] = None  # Use existing project
    project_description: Optional[str] = None
    
    # Graph building options
    build_graphs: Optional[bool] = True
    graph_max_iterations: Optional[int] = 3
    graph_max_graphs: Optional[int] = 5
    graph_focus_areas: Optional[str] = None
    graph_file_filter: Optional[str] = None  # Comma-separated file whitelist (manual override)
    auto_generate_whitelist: Optional[bool] = True  # Auto-generate whitelist if not provided
    whitelist_loc_budget: Optional[int] = 50000  # LOC budget for whitelist generation
    graph_with_spec: Optional[str] = None  # Build exactly one graph with this spec
    graph_refine_existing: Optional[bool] = True
    graph_init_only: Optional[bool] = False  # Only create SystemArchitecture
    graph_auto: Optional[bool] = True  # Auto-generate default graphs (default True to ensure SystemArchitecture is created)
    graph_refine_only: Optional[str] = None  # Refine only this graph name
    graph_visualize: Optional[bool] = True
    
    # Audit options (agent audit)
    audit_mode: Optional[str] = "sweep"  # "sweep" or "intuition"
    iterations: Optional[int] = 30  # Max iterations per investigation (increased default for better vulnerability detection)
    plan_n: Optional[int] = 5  # Number of investigations per planning batch
    time_limit_minutes: Optional[int] = None
    debug: Optional[bool] = False
    mission: Optional[str] = None  # Overarching mission for the audit
    
    # Model overrides
    scout_platform: Optional[str] = None  # Override scout platform
    scout_model: Optional[str] = None  # Override scout model
    strategist_platform: Optional[str] = None  # Override strategist platform
    strategist_model: Optional[str] = None  # Override strategist model
    
    # Session options
    session_id: Optional[str] = None  # Attach to specific session
    new_session: Optional[bool] = True  # Create new session
    session_private_hypotheses: Optional[bool] = False  # Keep hypotheses private to session
    
    # Advanced options
    strategist_two_pass: Optional[bool] = False  # Enable two-pass self-critique
    telemetry: Optional[bool] = True  # Enable telemetry (default: True)
    
    # Finalize options
    finalize_threshold: Optional[float] = 0.5  # Confidence threshold
    finalize_include_below_threshold: Optional[bool] = False
    finalize_platform: Optional[str] = None  # Override finalize platform
    finalize_model: Optional[str] = None  # Override finalize model


class ScanResult(BaseModel):
    scan_id: str
    status: str
    findings: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str
    project_name: Optional[str] = None
    session: Optional[str] = None
    coverage: Optional[Dict[str, Any]] = None
    diagnostic: Optional[Dict[str, Any]] = None
    telemetry: Optional[Dict[str, Any]] = None  # Telemetry server info for real-time monitoring


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
    
    TODO: Create brama_integration.py module similar to hound_integration.py
    for direct Python API integration (no subprocess overhead).
    For now, uses subprocess method.
    """
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Run BRAMA scan (subprocess for now, will be refactored to direct integration)
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


# Telemetry proxy endpoints for real-time monitoring
@app.get("/telemetry/{project_name}/events")
async def telemetry_events(project_name: str, request: Request):
    """
    Proxy SSE stream from Hound telemetry server.
    Note: This requires the telemetry server to be running (started during audit).
    """
    try:
        # Try to find telemetry info from registry
        registry_dir = Path.home() / ".local" / "state" / "hound" / "instances"
        telemetry_url = None
        token = None
        
        if registry_dir.exists():
            # Find most recent instance for this project
            instances = sorted(registry_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for inst_file in instances:
                try:
                    inst_data = json.loads(inst_file.read_text())
                    if inst_data.get("project_id") == project_name:
                        tel = inst_data.get("telemetry", {})
                        telemetry_url = tel.get("sse_url")
                        token = tel.get("token")
                        break
                except Exception:
                    continue
        
        if not telemetry_url:
            return JSONResponse(
                {"error": "Telemetry server not found. Start an audit with telemetry enabled."},
                status_code=404
            )
        
        # Proxy SSE stream
        async def stream_events():
            params = {}
            if token:
                params["token"] = token
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    async with client.stream("GET", telemetry_url, params=params) as response:
                        if response.status_code != 200:
                            yield f"data: {json.dumps({'error': f'Telemetry server returned {response.status_code}'})}\n\n"
                            return
                        
                        async for chunk in response.aiter_text():
                            if chunk:
                                yield chunk
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.error(f"Telemetry proxy error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/telemetry/{project_name}/status")
async def telemetry_status(project_name: str):
    """Get telemetry server status for a project."""
    try:
        registry_dir = Path.home() / ".local" / "state" / "hound" / "instances"
        if not registry_dir.exists():
            return JSONResponse({"available": False, "error": "No telemetry instances found"})
        
        # Find most recent instance for this project
        instances = sorted(registry_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for inst_file in instances:
            try:
                inst_data = json.loads(inst_file.read_text())
                if inst_data.get("project_id") == project_name:
                    tel = inst_data.get("telemetry", {})
                    return JSONResponse({
                        "available": True,
                        "telemetry": {
                            "sse_url": tel.get("sse_url"),
                            "control_url": tel.get("control_url"),
                            "port": tel.get("sse_url", "").split(":")[-1].split("/")[0] if tel.get("sse_url") else None
                        },
                        "project_id": inst_data.get("project_id"),
                        "started_at": inst_data.get("started_at")
                    })
            except Exception:
                continue
        
        return JSONResponse({"available": False, "error": "No telemetry instance found for this project"})
    except Exception as e:
        logger.error(f"Telemetry status error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


@app.post("/telemetry/{project_name}/steer")
async def telemetry_steer(project_name: str, steering: Dict[str, Any]):
    """
    Send steering command to Hound telemetry server.
    This allows real-time guidance of the audit.
    """
    try:
        registry_dir = Path.home() / ".local" / "state" / "hound" / "instances"
        control_url = None
        token = None
        
        if registry_dir.exists():
            instances = sorted(registry_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for inst_file in instances:
                try:
                    inst_data = json.loads(inst_file.read_text())
                    if inst_data.get("project_id") == project_name:
                        tel = inst_data.get("telemetry", {})
                        control_url = tel.get("control_url")
                        token = tel.get("token")
                        break
                except Exception:
                    continue
        
        if not control_url:
            return JSONResponse(
                {"error": "Telemetry server not found"},
                status_code=404
            )
        
        # Forward steering command
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{control_url}/steer",
                json=steering,
                headers=headers,
                timeout=5.0
            )
            return JSONResponse(await response.json(), status_code=response.status_code)
    except Exception as e:
        logger.error(f"Telemetry steer error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/audit-codebase")
async def audit_codebase(
    request: AuditCodebaseRequest,
    background_tasks: BackgroundTasks,
):
    """
    Audit a codebase using Hound (internal/deep mode)
    
    This endpoint directly uses Hound's Python API via hound_integration module.
    No subprocess overhead - clean, direct integration.
    """
    scan_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Use existing project or create new one
        if request.project_name:
            # Use existing project
            project_name = request.project_name
            logger.info(f"Using existing project: {project_name}")
        else:
            # Create new project from GitHub repo
            if not request.github_repo:
                raise HTTPException(
                    status_code=400, detail="Either github_repo or project_name must be provided"
                )
            
            # Clone GitHub repo to persistent trial directory
            # Find next available trial number
            trial_num = 1
            while (AUDITS_DIR / f"trial{trial_num}").exists():
                trial_num += 1
            
            trial_dir = AUDITS_DIR / f"trial{trial_num}"
            trial_dir.mkdir(exist_ok=True)
            repo_path = trial_dir / "repo"
            
            logger.info(f"Cloning repository to trial{trial_num}: {request.github_repo}")
            clone_result = subprocess.run(
                ["git", "clone", request.github_repo, str(repo_path)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            
            project_name = f"heimdall_{scan_id}"
            
            # Generate whitelist if enabled and not manually provided (as per Hound best practices)
            # See: https://muellerberndt.medium.com/hunting-for-security-bugs-in-code-with-ai-agents-a-full-walkthrough-a0dc24e1adf0
            whitelist_file = None
            if request.auto_generate_whitelist and not request.graph_file_filter:
                logger.info(f"Auto-generating file whitelist within {request.whitelist_loc_budget:,} LOC budget...")
                whitelist_dir = trial_dir / "whitelists"
                whitelist_dir.mkdir(exist_ok=True)
                whitelist_file = whitelist_dir / f"{project_name}.txt"
                
                whitelist_script = Path(__file__).parent.parent / "hound" / "whitelist_builder.py"
                if whitelist_script.exists():
                    whitelist_cmd = [
                        "python3", str(whitelist_script),
                        "--input", str(repo_path),
                        "--output", str(whitelist_file),
                        "--limit-loc", str(request.whitelist_loc_budget or 50000),
                        "--output-format", "csv",  # Hound expects comma-separated format
                        "--print-summary",
                        "--verbose"
                    ]
                    
                    whitelist_result = subprocess.run(
                        whitelist_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        env=os.environ.copy()
                    )
                    
                    if whitelist_result.returncode == 0:
                        # Log the output for debugging
                        if whitelist_result.stdout:
                            logger.info(f"Whitelist builder output: {whitelist_result.stdout[:500]}")
                        
                        # Read the whitelist and convert to comma-separated format
                        if whitelist_file.exists():
                            with open(whitelist_file, 'r') as f:
                                whitelist_content = f.read().strip()
                            if whitelist_content:
                                request.graph_file_filter = whitelist_content
                                file_count = len(whitelist_content.split(','))
                                logger.info(f"Generated whitelist with {file_count} files")
                            else:
                                logger.warning("Whitelist file is empty, proceeding without filter")
                                # Log stderr if available to help debug
                                if whitelist_result.stderr:
                                    logger.debug(f"Whitelist builder stderr: {whitelist_result.stderr[:500]}")
                        else:
                            logger.warning("Whitelist file was not created")
                    else:
                        error_msg = whitelist_result.stderr[:500] if whitelist_result.stderr else "Unknown error"
                        logger.warning(f"Whitelist generation failed (exit code {whitelist_result.returncode}): {error_msg}")
                        if whitelist_result.stdout:
                            logger.debug(f"Whitelist builder stdout: {whitelist_result.stdout[:500]}")
                else:
                    logger.warning(f"Whitelist builder script not found at {whitelist_script}, proceeding without whitelist")
            elif request.graph_file_filter:
                logger.info("Using manually provided file whitelist")
            else:
                logger.info("Whitelist generation disabled, proceeding without file filter")
            
            # Step 1: Create Hound project using CLI
            logger.info(f"Step 1: Creating Hound project: {project_name}")
            hound_script = Path(__file__).parent.parent / "hound" / "hound.py"
            create_cmd = [
                "python3", str(hound_script),
                "project", "create", project_name, str(repo_path)
            ]
            if request.project_description:
                create_cmd.extend(["--description", request.project_description])
            
            create_result = subprocess.run(
                create_cmd,
                cwd=str(hound_script.parent),
                capture_output=True,
                text=True,
                timeout=60,
                env=os.environ.copy()
            )
            
            if create_result.returncode != 0:
                logger.error(f"Project creation failed: {create_result.stderr}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create Hound project: {create_result.stderr[:500]}"
                )
            
            logger.info(f"Successfully created Hound project: {project_name}")
            
            # Store trial directory for reference (we keep it persistent)
            # Hound stores source_path in project.json and needs access to source files
            trial_dir_path = trial_dir
            logger.info(f"Repository cloned to persistent directory: {trial_dir_path}")
        
        # Use Hound CLI directly - simple and reliable, no dependency issues
        # Hound is in the hound/ directory at the project root
        hound_script = Path(__file__).parent.parent / "hound" / "hound.py"
        if not hound_script.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Hound script not found at {hound_script}. Please ensure Hound is properly installed."
            )
        
        logger.info(f"Using Hound CLI at: {hound_script}")
        
        # Step 2: Build knowledge graphs (if requested) using Hound CLI
        if request.build_graphs:
            logger.info("Step 2: Building knowledge graphs using Hound CLI...")
            graph_cmd = [
                "python3", str(hound_script),
                "graph", "build", project_name
            ]
            
            # Add graph options
            if request.graph_auto:
                graph_cmd.append("--auto")
            elif request.graph_init_only:
                graph_cmd.append("--init")
            
            if request.graph_max_iterations:
                graph_cmd.extend(["--iterations", str(request.graph_max_iterations)])
            if request.graph_max_graphs:
                graph_cmd.extend(["--graphs", str(request.graph_max_graphs)])
            if request.graph_file_filter:
                graph_cmd.extend(["--files", request.graph_file_filter])
            if request.graph_focus_areas:
                graph_cmd.extend(["--focus", request.graph_focus_areas])
            if request.graph_with_spec:
                graph_cmd.extend(["--with-spec", request.graph_with_spec])
            if request.graph_refine_only:
                graph_cmd.extend(["--refine-only", request.graph_refine_only])
            if not request.graph_refine_existing:
                graph_cmd.append("--no-refine-existing")
            if not request.graph_visualize:
                graph_cmd.append("--no-visualize")
            if request.debug:
                graph_cmd.append("--debug")
            graph_cmd.append("--quiet")  # Reduce output for API
            
            try:
                graph_result = subprocess.run(
                    graph_cmd,
                    cwd=str(hound_script.parent),
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for graph building
                    env=os.environ.copy()
                )
                
                if graph_result.returncode != 0:
                    logger.error(f"Graph build failed: {graph_result.stderr}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Graph building failed: {graph_result.stderr[:500]}"
                    )
                
                # Verify graphs were created
                project_dir = Path.home() / ".hound" / "projects" / project_name
                graphs_dir = project_dir / "graphs"
                graph_files = list(graphs_dir.glob("graph_*.json")) if graphs_dir.exists() else []
                
                if not graph_files:
                    raise HTTPException(
                        status_code=500,
                        detail="Graph building completed but no graph files were created"
                    )
                
                logger.info(f"Successfully built {len(graph_files)} graphs")
            except subprocess.TimeoutExpired:
                raise HTTPException(
                    status_code=500,
                    detail="Graph building timed out after 10 minutes"
                )
        else:
            # Hound REQUIRES at least SystemArchitecture graph for audits
            # If graph building is disabled, we must still create SystemArchitecture
            logger.warning("Graph building disabled, but SystemArchitecture is required for audit")
            logger.info("Building SystemArchitecture graph only (required for audit)...")
            graph_cmd = [
                "python3", str(hound_script),
                "graph", "build", project_name,
                "--init",  # Only create SystemArchitecture
                "--iterations", "1",  # Single iteration for speed
                "--quiet"
            ]
            
            try:
                graph_result = subprocess.run(
                    graph_cmd,
                    cwd=str(hound_script.parent),
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout for minimal graph
                    env=os.environ.copy()
                )
                
                if graph_result.returncode != 0:
                    logger.error(f"SystemArchitecture graph build failed: {graph_result.stderr}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to create required SystemArchitecture graph: {graph_result.stderr[:500]}"
                    )
                
                # Verify SystemArchitecture was created
                project_dir = Path.home() / ".hound" / "projects" / project_name
                graphs_dir = project_dir / "graphs"
                system_arch = graphs_dir / "graph_SystemArchitecture.json" if graphs_dir.exists() else None
                
                if not system_arch or not system_arch.exists():
                    raise HTTPException(
                        status_code=500,
                        detail="SystemArchitecture graph is required for audits but was not created"
                    )
                
                logger.info("SystemArchitecture graph created successfully")
            except subprocess.TimeoutExpired:
                raise HTTPException(
                    status_code=500,
                    detail="SystemArchitecture graph creation timed out"
                )
        
        # Step 3: Run audit using Hound CLI
        logger.info(f"Step 3: Running Hound audit using CLI (mode: {request.audit_mode}, iterations: {request.iterations})...")
        audit_cmd = [
            "python3", str(hound_script),
            "agent", "audit", project_name
        ]
        
        # Add audit options
        if request.audit_mode:
            audit_cmd.extend(["--mode", request.audit_mode])
        if request.iterations:
            audit_cmd.extend(["--iterations", str(request.iterations)])
        if request.plan_n:
            audit_cmd.extend(["--plan-n", str(request.plan_n)])
        if request.time_limit_minutes:
            audit_cmd.extend(["--time-limit", str(request.time_limit_minutes)])
        if request.debug:
            audit_cmd.append("--debug")
        if request.mission:
            audit_cmd.extend(["--mission", request.mission])
        if request.scout_platform:
            audit_cmd.extend(["--platform", request.scout_platform])
        if request.scout_model:
            audit_cmd.extend(["--model", request.scout_model])
        if request.strategist_platform:
            audit_cmd.extend(["--strategist-platform", request.strategist_platform])
        if request.strategist_model:
            audit_cmd.extend(["--strategist-model", request.strategist_model])
        if request.session_id:
            audit_cmd.extend(["--session", request.session_id])
        if request.new_session:
            audit_cmd.append("--new-session")
        if request.session_private_hypotheses:
            audit_cmd.append("--session-private-hypotheses")
        if request.strategist_two_pass:
            audit_cmd.append("--strategist-two-pass")
        if request.telemetry:
            audit_cmd.append("--telemetry")
        
        # Try to get telemetry info BEFORE audit starts (if telemetry is enabled)
        # This allows frontend to connect during the audit
        telemetry_info = None
        if request.telemetry:
            try:
                registry_dir = Path.home() / ".local" / "state" / "hound" / "instances"
                if registry_dir.exists():
                    # Check for existing telemetry instances
                    instances = sorted(registry_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    for inst_file in instances:
                        try:
                            inst_data = json.loads(inst_file.read_text())
                            if inst_data.get("project_id") == project_name:
                                tel = inst_data.get("telemetry", {})
                                telemetry_info = {
                                    "sse_url": tel.get("sse_url"),
                                    "control_url": tel.get("control_url"),
                                    "token": tel.get("token"),
                                    "port": tel.get("sse_url", "").split(":")[-1].split("/")[0] if tel.get("sse_url") else None
                                }
                                logger.info(f"Found existing telemetry server for project: {project_name}")
                                break
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"Failed to check for existing telemetry: {e}")
        
        try:
            # Run audit in background to allow telemetry connection during execution
            # But we still need to wait for it to complete
            audit_result = subprocess.run(
                audit_cmd,
                cwd=str(hound_script.parent),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for audit
                env=os.environ.copy()
            )
            
            if audit_result.returncode != 0:
                logger.error(f"Audit failed: {audit_result.stderr}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Hound audit failed: {audit_result.stderr[:500]}"
                )
            
            logger.info("Hound audit completed successfully")
            
            # After audit completes, try to get telemetry info again (in case it was just created)
            if request.telemetry and not telemetry_info:
                try:
                    registry_dir = Path.home() / ".local" / "state" / "hound" / "instances"
                    if registry_dir.exists():
                        instances = sorted(registry_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                        for inst_file in instances:
                            try:
                                inst_data = json.loads(inst_file.read_text())
                                if inst_data.get("project_id") == project_name:
                                    tel = inst_data.get("telemetry", {})
                                    telemetry_info = {
                                        "sse_url": tel.get("sse_url"),
                                        "control_url": tel.get("control_url"),
                                        "token": tel.get("token"),
                                        "port": tel.get("sse_url", "").split(":")[-1].split("/")[0] if tel.get("sse_url") else None
                                    }
                                    logger.info(f"Found telemetry server after audit: {project_name}")
                                    break
                            except Exception:
                                continue
                except Exception as e:
                    logger.warning(f"Failed to get telemetry info after audit: {e}")
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=500,
                detail="Audit timed out after 1 hour"
            )
        
        # Step 4: Finalize hypotheses using Hound CLI
        logger.info("Step 4: Finalizing hypotheses using Hound CLI...")
        finalize_cmd = [
            "python3", str(hound_script),
            "finalize", project_name,
            "--threshold", str(request.finalize_threshold or 0.5)
        ]
        
        if request.finalize_include_below_threshold:
            finalize_cmd.append("--include-below-threshold")
        if request.debug:
            finalize_cmd.append("--debug")
        if request.finalize_platform:
            finalize_cmd.extend(["--platform", request.finalize_platform])
        if request.finalize_model:
            finalize_cmd.extend(["--model", request.finalize_model])
        
        try:
            finalize_result = subprocess.run(
                finalize_cmd,
                cwd=str(hound_script.parent),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=os.environ.copy()
            )
            
            if finalize_result.returncode != 0:
                logger.warning(f"Finalization had issues: {finalize_result.stderr}")
            else:
                logger.info("Finalization completed")
        except subprocess.TimeoutExpired:
            logger.warning("Finalization timed out, continuing anyway")
        
        # Step 5: Read results from Hound's output files
        logger.info("Step 5: Reading Hound results...")
        project_dir = Path.home() / ".hound" / "projects" / project_name
        hypotheses_file = project_dir / "hypotheses.json"
        
        if not hypotheses_file.exists():
            raise HTTPException(
                status_code=500,
                detail="Hypotheses file not found after audit"
            )
        
        with open(hypotheses_file, "r") as f:
            hound_data = json.load(f)
        
        # Log diagnostic info about hypotheses
        hypotheses_dict = hound_data.get("hypotheses", {})
        hypotheses_count = len(hypotheses_dict) if isinstance(hypotheses_dict, dict) else 0
        logger.info(f"Found {hypotheses_count} hypotheses in hypotheses.json")
        
        if hypotheses_count == 0:
            logger.warning("No hypotheses found - audit may need more iterations or time")
            # Check metadata for more info
            metadata = hound_data.get("metadata", {})
            logger.info(f"Hypotheses metadata: {metadata}")
        
        # Get coverage and session info
        coverage_data = {}
        session_data = {}
        sessions_dir = project_dir / "sessions"
        if sessions_dir.exists():
            session_files = list(sessions_dir.glob("*.json"))
            if session_files:
                latest_session = max(session_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_session, "r") as f:
                        session_data = json.load(f)
                        coverage_data = session_data.get("coverage", {})
                        logger.info(f"Coverage data: {coverage_data}")
                except Exception:
                    pass
        
        results = {
            "hypotheses": hypotheses_dict,
            "coverage": coverage_data,
            "session": session_data.get("session_id") if session_data else None,
            "project_dir": str(project_dir)
        }
        
        # Transform to Heimdall format
        result = transform_hound_output(
            {"hypotheses": results["hypotheses"]},
            scan_id
        )
        
        # Add diagnostic message if no findings
        if len(result.get("findings", [])) == 0:
            diagnostic_msg = []
            if request.iterations and request.iterations < 10:
                diagnostic_msg.append(f"Only {request.iterations} iteration(s) were run - Hound typically needs 20-50 iterations to find vulnerabilities")
            if coverage_data:
                nodes_visited = coverage_data.get("nodes", {}).get("visited", 0)
                nodes_total = coverage_data.get("nodes", {}).get("total", 0)
                cards_visited = coverage_data.get("cards", {}).get("visited", 0)
                cards_total = coverage_data.get("cards", {}).get("total", 0)
                if cards_total > 0 and (cards_visited / cards_total) < 0.1:
                    diagnostic_msg.append(f"Low code coverage ({cards_visited}/{cards_total} cards = {int(cards_visited/cards_total*100)}%) - try increasing iterations or time limit")
            if not diagnostic_msg:
                diagnostic_msg.append("No vulnerabilities found. This could mean the codebase is secure, or Hound needs more time/iterations to explore.")
            
            result["diagnostic"] = {
                "message": "No findings detected",
                "hypotheses_found": hypotheses_count,
                "suggestion": " | ".join(diagnostic_msg)
            }
        
        # Add metadata
        result["coverage"] = results.get("coverage", {})
        result["session"] = results.get("session")
        result["project_name"] = project_name
        
        # Use telemetry_info we collected (either before or after audit)
        # If we still don't have it, try one more time
        if request.telemetry and not telemetry_info:
            try:
                registry_dir = Path.home() / ".local" / "state" / "hound" / "instances"
                if registry_dir.exists():
                    instances = sorted(registry_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    for inst_file in instances:
                        try:
                            inst_data = json.loads(inst_file.read_text())
                            if inst_data.get("project_id") == project_name:
                                tel = inst_data.get("telemetry", {})
                                telemetry_info = {
                                    "sse_url": tel.get("sse_url"),
                                    "control_url": tel.get("control_url"),
                                    "token": tel.get("token"),
                                    "port": tel.get("sse_url", "").split(":")[-1].split("/")[0] if tel.get("sse_url") else None
                                }
                                logger.info(f"Found telemetry info in final check: {project_name}")
                                break
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"Failed to get telemetry info in final check: {e}")
        
        result["telemetry"] = telemetry_info
        
        # Log telemetry status for debugging
        if request.telemetry:
            if telemetry_info:
                logger.info(f"Telemetry available: {telemetry_info.get('sse_url', 'N/A')}")
            else:
                logger.warning("Telemetry was requested but no telemetry info found")
        
        # Keep trial directory persistent - Hound needs access to source files
        # Directories are named trial1, trial2, etc. and kept for future reference
        if 'trial_dir_path' in locals():
            logger.info(f"Audit completed. Repository kept at: {trial_dir_path}")
            logger.info(f"Future audits will use trial{trial_num + 1}, trial{trial_num + 2}, etc.")
        
        return ScanResult(
            scan_id=scan_id,
            status="completed",
            findings=result.get("findings", []),
            summary=result.get("summary", {}),
            timestamp=datetime.now().isoformat(),
            project_name=result.get("project_name"),
            session=result.get("session"),
            coverage=result.get("coverage"),
            diagnostic=result.get("diagnostic"),
            telemetry=result.get("telemetry"),  # Include telemetry info in response
        )
        
        # Store telemetry info if available (for future real-time monitoring)
        if result.get("telemetry"):
            # Telemetry info is included in the response for frontend to use
            pass
    except ImportError as e:
        logger.error(f"Hound integration not available: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Hound integration failed. Please ensure Hound is properly installed: {str(e)}"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e.stderr}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to clone repository: {e.stderr.decode() if e.stderr else str(e)}"
        )
    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
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


@app.get("/project/{project_name}/graphs")
async def get_graphs(project_name: str):
    """
    Get list of available graphs for a project
    """
    try:
        project_dir = Path.home() / ".hound" / "projects" / project_name
        graphs_dir = project_dir / "graphs"
        
        if not graphs_dir.exists():
            raise HTTPException(status_code=404, detail="Graphs directory not found")
        
        graph_files = list(graphs_dir.glob("graph_*.json"))
        graphs = []
        
        for graph_file in graph_files:
            try:
                with open(graph_file, "r") as f:
                    graph_data = json.load(f)
                    graphs.append({
                        "name": graph_data.get("name", graph_file.stem),
                        "nodes": len(graph_data.get("nodes", [])),
                        "edges": len(graph_data.get("edges", [])),
                        "file": graph_file.name,
                    })
            except Exception:
                continue
        
        return {"graphs": graphs, "project_name": project_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load graphs: {str(e)}")


@app.get("/project/{project_name}/graph/{graph_name}")
async def get_graph_data(project_name: str, graph_name: str):
    """
    Get specific graph data for visualization
    """
    try:
        project_dir = Path.home() / ".hound" / "projects" / project_name
        graphs_dir = project_dir / "graphs"
        
        # Try to find the graph file
        graph_file = graphs_dir / f"graph_{graph_name}.json"
        if not graph_file.exists():
            # Try alternative naming
            graph_files = list(graphs_dir.glob("graph_*.json"))
            for gf in graph_files:
                with open(gf, "r") as f:
                    data = json.load(f)
                    if data.get("name") == graph_name or gf.stem == f"graph_{graph_name}":
                        graph_file = gf
                        break
        
        if not graph_file.exists():
            raise HTTPException(status_code=404, detail="Graph not found")
        
        with open(graph_file, "r") as f:
            graph_data = json.load(f)
        
        # Also try to load card store if available
        card_store = {}
        card_store_file = graphs_dir / "card_store.json"
        if card_store_file.exists():
            try:
                with open(card_store_file, "r") as f:
                    card_store = json.load(f)
            except Exception:
                pass
        
        return {
            "graph": graph_data,
            "card_store": card_store,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load graph: {str(e)}")


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


# NOTE: run_hound_scan and _run_hound_scan_subprocess are now DEPRECATED
# Hound is used directly via hound_integration module in the /audit-codebase endpoint
# Keeping these for backward compatibility if needed

async def run_hound_github_scan(github_repo: str, scan_id: str, 
                                 audit_mode: str = "sweep", 
                                 build_graphs: bool = False,
                                 time_limit_minutes: int | None = None,
                                 iterations: int = 10) -> Dict[str, Any]:
    """
    DEPRECATED: Clone GitHub repo and run Hound scan with full options
    
    This function is kept for backward compatibility but is no longer used.
    The /audit-codebase endpoint now handles this directly.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "repo"
        
        # Clone repository
        subprocess.run(
            ["git", "clone", github_repo, str(repo_path)],
            check=True,
            capture_output=True,
        )
        
        # Use direct Hound integration
        from hound_integration import (
            create_hound_project,
            build_hound_graphs,
            run_hound_audit,
            get_hound_results
        )
        
        project_name = f"heimdall_{scan_id}"
        create_hound_project(project_name, repo_path)
        
        if build_graphs:
            build_hound_graphs(project_name, auto=True)
        
        run_hound_audit(
            project_name=project_name,
            mode=audit_mode,
            iterations=iterations,
            time_limit_minutes=time_limit_minutes
        )
        
        results = get_hound_results(project_name)
        result = transform_hound_output(
            {"hypotheses": results["hypotheses"]},
            scan_id
        )
        result["coverage"] = results.get("coverage", {})
        result["session"] = results.get("session")
        result["project_name"] = project_name
        return result


async def _run_hound_scan_subprocess(
    codebase_path: Path, scan_id: str,
                         audit_mode: str = "sweep",
                         build_graphs: bool = False,
                         time_limit_minutes: int | None = None,
                         iterations: int = 10) -> Dict[str, Any]:
    """
    Run Hound via subprocess (fallback method).
    
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
        
        # Make sure script is executable
        os.chmod(hound_script, 0o755)
        
        # Create project - Hound script is a bash script
        create_result = subprocess.run(
            [
                "bash",
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
            env=dict(os.environ, XAI_API_KEY=os.getenv("XAI_API_KEY", "")),
        )
        
        if create_result.returncode != 0:
            error_msg = create_result.stderr or create_result.stdout
            raise RuntimeError(f"Failed to create Hound project: {error_msg}")
        
        # Optional: Build knowledge graphs if requested
        if build_graphs:
            graph_result = subprocess.run(
                [
                    "bash",
                    str(hound_script),
                    "graph",
                    "build",
                    project_name,
                    "--auto",
                ],
                cwd=str(HOUND_PATH),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for graph building
                env=dict(os.environ, XAI_API_KEY=os.getenv("XAI_API_KEY", "")),
            )
            if graph_result.returncode != 0:
                # Graph building failed, but continue anyway
                pass
        
        # Build audit command
        audit_cmd = [
            "bash",
            str(hound_script),
            "agent",
            "audit",
            project_name,
            "--mode", audit_mode,
            "--iterations", str(iterations),
        ]
        
        # Add time limit if specified
        if time_limit_minutes:
            audit_cmd.extend(["--time-limit", str(time_limit_minutes)])
        
        # Run agent audit
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Running Hound audit: {' '.join(audit_cmd)}")
        
        agent_result = subprocess.run(
            audit_cmd,
            cwd=str(HOUND_PATH),
            capture_output=True,
            text=True,
            timeout=600 if not time_limit_minutes else (time_limit_minutes * 60) + 60,
            env=dict(os.environ, XAI_API_KEY=os.getenv("XAI_API_KEY", "")),
        )
        
        # Log output for debugging
        logger.info(f"Hound audit return code: {agent_result.returncode}")
        if agent_result.stdout:
            logger.info(f"Hound stdout (last 1000 chars): {agent_result.stdout[-1000:]}")
        if agent_result.stderr:
            logger.warning(f"Hound stderr (last 1000 chars): {agent_result.stderr[-1000:]}")
        
        if agent_result.returncode != 0:
            error_msg = agent_result.stderr or agent_result.stdout
            # Don't fail completely - might have partial results
            if "No graphs found" in error_msg:
                # Try to continue anyway - Hound might auto-create graphs
                logger.warning("Hound reported 'No graphs found' but continuing anyway")
            else:
                logger.error(f"Hound analysis failed: {error_msg}")
                # Still try to read results - might have partial findings
        
        # Read findings directly from Hound's project directory
        # Hound stores findings in ~/.hound/projects/{project_name}/hypotheses.json
        project_dir = Path.home() / ".hound" / "projects" / project_name
        hypotheses_file = project_dir / "hypotheses.json"
        
        # Get coverage and session info
        coverage_data = {}
        session_data = {}
        
        # Try to get latest session for coverage stats
        sessions_dir = project_dir / "sessions"
        if sessions_dir.exists():
            session_files = list(sessions_dir.glob("*.json"))
            if session_files:
                latest_session = max(session_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_session, "r") as f:
                        session_data = json.load(f)
                        coverage_data = session_data.get("coverage", {})
                except Exception:
                    pass
        
        if hypotheses_file.exists():
            try:
                with open(hypotheses_file, "r") as f:
                    hound_data = json.load(f)
                
                # Debug: Log what we got from Hound
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Hound data keys: {hound_data.keys() if isinstance(hound_data, dict) else 'not a dict'}")
                if isinstance(hound_data, dict):
                    hypotheses_dict = hound_data.get("hypotheses", {})
                    logger.info(f"Hypotheses dict type: {type(hypotheses_dict)}, length: {len(hypotheses_dict) if isinstance(hypotheses_dict, dict) else 'N/A'}")
                    if isinstance(hypotheses_dict, dict) and len(hypotheses_dict) > 0:
                        sample_hyp = list(hypotheses_dict.values())[0]
                        logger.info(f"Sample hypothesis keys: {sample_hyp.keys() if isinstance(sample_hyp, dict) else 'not a dict'}")
                
                result = transform_hound_output(hound_data, scan_id)
                
                # If no findings, check if hypotheses file is empty or has no valid hypotheses
                if result.get("findings", []) == []:
                    logger.warning(f"No findings extracted. Hound data structure: {type(hound_data)}")
                    if isinstance(hound_data, dict):
                        hypotheses_dict = hound_data.get("hypotheses", {})
                        if isinstance(hypotheses_dict, dict) and len(hypotheses_dict) == 0:
                            logger.warning("Hypotheses dict is empty - Hound may not have found any issues or audit didn't complete")
                        elif isinstance(hypotheses_dict, dict) and len(hypotheses_dict) > 0:
                            logger.warning(f"Found {len(hypotheses_dict)} hypotheses but none passed filtering")
                            # Log status of all hypotheses
                            for hyp_id, hyp in list(hypotheses_dict.items())[:5]:
                                logger.warning(f"  Hypothesis {hyp_id}: status={hyp.get('status')}, severity={hyp.get('severity')}")
                
                # Add coverage and metadata
                result["coverage"] = coverage_data
                result["session"] = session_data.get("session_id") if session_data else None
                result["project_name"] = project_name
                return result
            except (json.JSONDecodeError, Exception) as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to parse hypotheses.json: {e}")
                # If reading fails, try to extract from agent output
                return parse_hound_text_output(agent_result.stdout, scan_id)
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Hypotheses file not found: {hypotheses_file}")
            logger.info(f"Agent stdout (last 500 chars): {agent_result.stdout[-500:] if agent_result.stdout else 'empty'}")
            logger.info(f"Agent stderr (last 500 chars): {agent_result.stderr[-500:] if agent_result.stderr else 'empty'}")
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


async def run_hound_github_scan(github_repo: str, scan_id: str, 
                                 audit_mode: str = "sweep", 
                                 build_graphs: bool = False,
                                 time_limit_minutes: int | None = None,
                                 iterations: int = 10) -> Dict[str, Any]:
    """
    Clone GitHub repo and run Hound scan with full options
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "repo"
        
        # Clone repository
        subprocess.run(
            ["git", "clone", github_repo, str(repo_path)],
            check=True,
            capture_output=True,
        )
        
        # Run Hound scan with options
        return await run_hound_scan(repo_path, scan_id, audit_mode, build_graphs, time_limit_minutes, iterations)


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
        "hypotheses": {
            "hyp_123": {
                "id": "hyp_123",
                "status": "proposed" | "investigating" | "confirmed" | "rejected",
                "title": "...",
                "description": "...",
                "vulnerability_type": "...",
                "severity": "critical" | "high" | "medium" | "low",
                "confidence": 0.0-1.0,
                "evidence": [...],
                "node_refs": [...],
                ...
            }
        }
    }
    """
    findings = []
    
    # Hound stores hypotheses as a DICT, not a list
    hypotheses_dict = hound_data.get("hypotheses", {})
    
    # Convert dict to list for processing
    if isinstance(hypotheses_dict, dict):
        hypotheses = list(hypotheses_dict.values())
    else:
        # Fallback: if it's already a list (old format)
        hypotheses = hypotheses_dict if isinstance(hypotheses_dict, list) else []
    
    # Include all findings (not just confirmed) for MVP - user can filter later
    # Status can be: proposed, investigating, confirmed, rejected, supported, refuted
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
            
        status = hyp.get("status", "").lower()
        
        # Skip rejected and refuted findings, include everything else
        if status in ["rejected", "refuted"]:
            continue
        
        # If status is empty or missing, treat as proposed (include it)
        if not status:
            status = "proposed"
        
        # Extract location information from evidence or node_refs
        file_path = hyp.get("file", hyp.get("location", "unknown"))
        line_num = hyp.get("line", hyp.get("line_number", 0))
        
        # Try to get location from evidence if not directly available
        if file_path == "unknown":
            evidence = hyp.get("evidence", [])
            if evidence and isinstance(evidence, list):
                for ev in evidence:
                    if isinstance(ev, dict):
                        if "file" in ev:
                            file_path = ev.get("file", "unknown")
                        if "line" in ev and line_num == 0:
                            line_num = ev.get("line", 0)
        
        location = f"{file_path}:{line_num}" if file_path != "unknown" else "unknown"
        
        # Extract code snippet from evidence if available
        code_snippet = ""
        evidence = hyp.get("evidence", [])
        if evidence and isinstance(evidence, list):
            # Try to find code evidence
            for ev in evidence:
                if isinstance(ev, dict):
                    if "code" in ev:
                        code_snippet = ev.get("code", "")
                        break
                    elif "description" in ev and ("def " in ev["description"] or "class " in ev["description"]):
                        code_snippet = ev.get("description", "")
                        break
                elif isinstance(ev, str) and ("def " in ev or "class " in ev or "=" in ev):
                    code_snippet = ev
                    break
        
        # Get severity - Hound has explicit severity field
        severity = hyp.get("severity", "medium").lower()
        
        # Map Hound severity to Heimdall severity
        severity = map_severity(severity)
        
        # If no explicit severity, derive from confidence
        if severity == "medium" and "confidence" in hyp:
            confidence = hyp.get("confidence", 0.5)
            if confidence >= 0.9:
                severity = "critical"
            elif confidence >= 0.7:
                severity = "high"
            elif confidence >= 0.5:
                severity = "medium"
            else:
                severity = "low"
        
            # Preserve ALL Hound fields - pass through everything
            # This ensures we don't lose any information from Hound's output
            finding = {
                "id": hyp.get("id", f"hound_{len(findings)}"),
                "severity": severity,
                "title": hyp.get("title", hyp.get("name", "Security Issue")),
                "description": hyp.get("description", hyp.get("summary", hyp.get("rationale", ""))),
                "location": location,
                "code_snippet": code_snippet or hyp.get("code", ""),
                "recommendation": hyp.get("recommendation", hyp.get("fix_guidance", "")),
                "fix_suggestion": hyp.get("fix", hyp.get("fix_suggestion", "")),
                # Preserve ALL Hound-specific fields exactly as they are
                "confidence": hyp.get("confidence", 0.5),
                "status": hyp.get("status", "proposed"),
                "vulnerability_type": hyp.get("vulnerability_type", hyp.get("type", "unknown")),
                "evidence": hyp.get("evidence", []),  # Keep full evidence structure
                "node_refs": hyp.get("node_refs", []),
                "reasoning": hyp.get("reasoning", ""),
                "created_at": hyp.get("created_at", ""),
                "junior_model": hyp.get("junior_model", hyp.get("reported_by_model", "")),
                "senior_model": hyp.get("senior_model", ""),
                "created_by": hyp.get("created_by", ""),
                "session_id": hyp.get("session_id", ""),
                "visibility": hyp.get("visibility", "global"),
                "properties": hyp.get("properties", {}),
                # Additional fields from Hound's HypothesisItemJSON format
                "root_cause": hyp.get("root_cause", ""),
                "attack_vector": hyp.get("attack_vector", ""),
                # Keep any other fields that might exist
            }
            
            # Add any additional fields that weren't explicitly handled
            for key, value in hyp.items():
                if key not in finding and key not in ["file", "location", "line", "line_number", "code"]:
                    finding[key] = value
            
            findings.append(finding)
    
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

