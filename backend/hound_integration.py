"""
Direct Python integration with Hound - uses Hound as a library instead of subprocess.

This module provides a clean Python API for using Hound directly, without subprocess overhead.
When a user chooses "Audit Codebase" in the frontend, this module is used directly.

Benefits:
- Faster execution (no subprocess overhead)
- Better error handling (Python exceptions)
- Direct access to Hound's internal state
- Cleaner code architecture
"""

import sys
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from backend/.env so XAI_API_KEY is available to Hound
# This ensures Hound can access the API key when it reads config.yaml
# Hound's config.yaml specifies: xai: api_key_env: XAI_API_KEY
# So we need to make sure XAI_API_KEY is in the environment before importing Hound modules
_backend_dir = Path(__file__).parent
_env_file = _backend_dir / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=True)  # override=True ensures backend/.env takes precedence
    
    # Verify XAI_API_KEY is available
    if not os.getenv("XAI_API_KEY"):
        logger_temp = logging.getLogger(__name__)
        logger_temp.warning(
            "⚠️  XAI_API_KEY not found in backend/.env - Hound will fail to authenticate. "
            "Please add XAI_API_KEY=your_key to backend/.env"
        )
    else:
        # Set it explicitly to ensure it's available to child processes
        os.environ["XAI_API_KEY"] = os.getenv("XAI_API_KEY")

# Add Hound to path (similar to how hound.py does it)
_HOUND_BASE = Path(__file__).parent.parent / "agents" / "hound"
_HOUND_BASE_STR = str(_HOUND_BASE)

if _HOUND_BASE.exists():
    if _HOUND_BASE_STR not in sys.path:
        sys.path.insert(0, _HOUND_BASE_STR)
    
    # Handle Hound's llm module namespace hack
    try:
        import types
        if 'llm' not in sys.modules:
            _LLM_DIR = _HOUND_BASE / "llm"
            m = types.ModuleType('llm')
            m.__path__ = [str(_LLM_DIR)]
            sys.modules['llm'] = m
    except Exception:
        pass

logger = logging.getLogger(__name__)


def create_hound_project(project_name: str, source_path: Path, description: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a Hound project directly using ProjectManager.
    
    Returns project config dict.
    """
    try:
        from commands.project import ProjectManager
        
        manager = ProjectManager()
        config = manager.create_project(
            name=project_name,
            source_path=str(source_path),
            description=description,
            auto_name=False
        )
        
        logger.info(f"Created Hound project: {project_name} at {source_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to create Hound project: {e}")
        raise RuntimeError(f"Failed to create Hound project: {str(e)}")


def build_hound_graphs(
    project_name: str,
    auto: bool = True,
    file_whitelist: Optional[str] = None,
    max_iterations: int = 3,
    max_graphs: int = 5,
    focus_areas: Optional[str] = None,
    with_spec: Optional[str] = None,
    refine_existing: bool = True,
    init_only: bool = False,
    refine_only: Optional[str] = None,
    visualize: bool = True,
    debug: bool = False,
    quiet: bool = False
) -> bool:
    """
    Build knowledge graphs for a Hound project.
    
    Following the workflow from: https://muellerberndt.medium.com/hunting-for-security-bugs-in-code-with-ai-agents-a-full-walkthrough-a0dc24e1adf0
    
    Args:
        project_name: Name of the Hound project
        auto: Use auto mode to generate default set of graphs (recommended)
        file_whitelist: Comma-separated list of files to include (recommended for large repos)
    
    Returns True if successful, False otherwise.
    """
    try:
        from commands.graph import build as graph_build
        from utils.config_loader import load_config
        import click
        
        # Load config
        config = load_config()
        
        # Build graphs using Hound's graph build command with all options
        # IMPORTANT: If auto is False and init_only is False, we need to ensure SystemArchitecture is built
        # because AgentRunner.initialize() requires SystemArchitecture graph to exist
        ctx = click.Context(graph_build)
        
        # If neither auto nor init_only is set, default to init to create SystemArchitecture
        # This ensures the audit can run
        should_init = init_only
        if not auto and not init_only:
            # Check if SystemArchitecture already exists
            project_dir = Path.home() / ".hound" / "projects" / project_name
            graphs_dir = project_dir / "graphs"
            sys_arch = graphs_dir / "graph_SystemArchitecture.json"
            if not sys_arch.exists():
                logger.info("SystemArchitecture graph not found - will initialize it")
                should_init = True
        
        params = {
            'project_id': project_name,
            'auto': auto,
            'max_iterations': max_iterations,
            'max_graphs': max_graphs,
            'visualize': visualize,
            'debug': debug,
            'quiet': quiet,
            'refine_existing': refine_existing,
            'init': should_init,
            'refine_only': refine_only,
        }
        
        # Add optional parameters
        if file_whitelist:
            params['file_filter'] = file_whitelist
        if focus_areas:
            params['focus_areas'] = focus_areas
        if with_spec:
            params['with_spec'] = with_spec
        
        ctx.params = params
        
        try:
            graph_build.invoke(ctx)
            logger.info(f"Graph build command completed for project: {project_name}")
            
            # Verify graphs were actually created
            project_dir = Path.home() / ".hound" / "projects" / project_name
            graphs_dir = project_dir / "graphs"
            
            if not graphs_dir.exists():
                logger.error(f"Graphs directory was not created: {graphs_dir}")
                return False
            
            graph_files = list(graphs_dir.glob("graph_*.json"))
            if not graph_files:
                logger.error(f"No graph files found in {graphs_dir} after build")
                return False
            
            # Check specifically for SystemArchitecture (required by AgentRunner)
            sys_arch = graphs_dir / "graph_SystemArchitecture.json"
            if not sys_arch.exists():
                logger.warning(f"SystemArchitecture graph not found, but found {len(graph_files)} other graphs")
                # This is a warning, not a failure - we'll try to continue
            
            logger.info(f"Successfully created {len(graph_files)} graph files")
            return True
        except SystemExit as e:
            if e.code == 0:
                # Even if exit code is 0, verify graphs exist
                project_dir = Path.home() / ".hound" / "projects" / project_name
                graphs_dir = project_dir / "graphs"
                graph_files = list(graphs_dir.glob("graph_*.json")) if graphs_dir.exists() else []
                if graph_files:
                    logger.info(f"Graph build exited with code 0, found {len(graph_files)} graphs")
                    return True
                else:
                    logger.error("Graph build exited with code 0 but no graphs found")
                    return False
            logger.error(f"Graph build exited with code {e.code}")
            return False
    except Exception as e:
        logger.error(f"Failed to build graphs: {e}", exc_info=True)
        return False


def run_hound_audit(
    project_name: str,
    mode: str = "sweep",
    iterations: int = 20,
    time_limit_minutes: Optional[int] = None,
    config_path: Optional[Path] = None,
    enable_telemetry: bool = True,
    # Additional CLI options
    plan_n: int = 5,
    debug: bool = False,
    mission: Optional[str] = None,
    scout_platform: Optional[str] = None,
    scout_model: Optional[str] = None,
    strategist_platform: Optional[str] = None,
    strategist_model: Optional[str] = None,
    session_id: Optional[str] = None,
    new_session: bool = True,
    session_private_hypotheses: bool = False,
    strategist_two_pass: bool = False
) -> Dict[str, Any]:
    """
    Run Hound audit directly using AgentRunner.
    
    Args:
        project_name: Name of the Hound project
        mode: Audit mode ("sweep" or "intuition")
        iterations: Number of iterations
        time_limit_minutes: Optional time limit
        config_path: Optional config file path
        enable_telemetry: Whether to start telemetry server for real-time monitoring
    
    Returns dict with audit results, metadata, and telemetry info.
    """
    try:
        from commands.agent import AgentRunner
        from pathlib import Path
        
        project_dir = Path.home() / ".hound" / "projects" / project_name
        
        # Start telemetry server if enabled
        telemetry_server = None
        telemetry_info = None
        if enable_telemetry:
            try:
                from telemetry import TelemetryServer
                telemetry_server = TelemetryServer(str(project_name), project_dir)
                telemetry_server.start()
                logger.info(f"Telemetry server started on port {telemetry_server.httpd.server_address[1] if telemetry_server.httpd else 'unknown'}")
                
                # Get telemetry info for return
                if telemetry_server.httpd:
                    port = telemetry_server.httpd.server_address[1]
                    telemetry_info = {
                        "sse_url": f"http://127.0.0.1:{port}/events",
                        "control_url": f"http://127.0.0.1:{port}",
                        "token": telemetry_server.token,
                        "port": port
                    }
                    # Emit startup event
                    telemetry_server.publish({'type': 'status', 'message': 'audit session started', 'iteration': 0})
            except Exception as e:
                logger.warning(f"Failed to start telemetry server (continuing without it): {e}")
                telemetry_server = None
        
        # Create AgentRunner instance with all CLI options
        runner = AgentRunner(
            project_id=project_name,
            config_path=config_path,
            iterations=iterations,
            time_limit_minutes=time_limit_minutes,
            debug=debug,
            platform=scout_platform,
            model=scout_model,
            session=session_id,
            new_session=new_session,
            mode=mode
        )
        
        # Set additional options if available
        if mission:
            try:
                runner.mission = mission
            except Exception:
                pass
        
        # Note: Some options (strategist_platform, strategist_model, session_private_hypotheses, 
        # strategist_two_pass) may need to be set via config.yaml or on the runner instance
        # For now, we log them for visibility
        if strategist_platform or strategist_model:
            logger.info(f"Strategist overrides: platform={strategist_platform}, model={strategist_model}")
            # These would typically be set via config.yaml or runner attributes if supported
        
        if session_private_hypotheses:
            logger.info("Session private hypotheses enabled")
            # This may need to be set on the session tracker if supported
        
        if strategist_two_pass:
            logger.info("Strategist two-pass enabled")
            # This may need to be set on the strategist if supported
        
        # Initialize the runner
        try:
            init_result = runner.initialize()
            if not init_result:
                # Get more details about why initialization failed
                project_dir = Path.home() / ".hound" / "projects" / project_name
                graphs_dir = project_dir / "graphs"
                
                error_details = []
                if not project_dir.exists():
                    error_details.append(f"Project directory does not exist: {project_dir}")
                elif not graphs_dir.exists():
                    error_details.append(f"Graphs directory does not exist: {graphs_dir}")
                else:
                    graph_files = list(graphs_dir.glob("graph_*.json"))
                    if not graph_files:
                        error_details.append(f"No graph files found in {graphs_dir}")
                    else:
                        error_details.append(f"Found {len(graph_files)} graph files but initialization still failed")
                
                error_msg = "Failed to initialize Hound agent runner. " + "; ".join(error_details)
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            logger.error(f"Exception during runner initialization: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Hound agent runner: {str(e)}")
        
        # Inject telemetry publisher if telemetry is enabled
        if telemetry_server:
            runner._telemetry_publish = telemetry_server.publish  # type: ignore[attr-defined]
            logger.info("Telemetry publishing enabled for audit")
        
        # Run the audit
        logger.info(f"Running Hound audit in {mode} mode with {iterations} iterations, plan_n={plan_n}...")
        runner.run(plan_n=plan_n)  # plan_n is investigations per planning batch
        
        # Stop telemetry server
        if telemetry_server:
            try:
                telemetry_server.stop()
                logger.info("Telemetry server stopped")
            except Exception:
                pass
        
        return {
            "project_dir": str(project_dir),
            "project_name": project_name,
            "status": "completed",
            "telemetry": telemetry_info
        }
        
    except Exception as e:
        logger.error(f"Hound audit failed: {e}")
        # Ensure telemetry is stopped on error
        if 'telemetry_server' in locals() and telemetry_server:
            try:
                telemetry_server.stop()
            except Exception:
                pass
        raise RuntimeError(f"Hound audit failed: {str(e)}")


def finalize_hound_hypotheses(
    project_name: str,
    threshold: float = 0.5,
    include_below_threshold: bool = False,
    debug: bool = False,
    platform: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Finalize hypotheses by reviewing and confirming/rejecting high-confidence findings.
    
    Following the workflow from: https://muellerberndt.medium.com/hunting-for-security-bugs-in-code-with-ai-agents-a-full-walkthrough-a0dc24e1adf0
    
    This step reviews hypotheses above the confidence threshold and confirms/rejects them
    with full source code context.
    
    Args:
        project_name: Name of the Hound project
        threshold: Confidence threshold for review (default: 0.5)
        include_below_threshold: Also review hypotheses below threshold
    
    Returns dict with finalization results.
    """
    try:
        from commands.finalize import finalize as finalize_command
        import click
        
        ctx = click.Context(finalize_command)
        ctx.params = {
            'project_name': project_name,
            'threshold': threshold,
            'include_below_threshold': include_below_threshold,
            'debug': debug,
            'platform': platform,
            'model': model
        }
        
        try:
            finalize_command.invoke(ctx)
            logger.info(f"Finalized hypotheses for project: {project_name}")
            return {"status": "completed"}
        except SystemExit as e:
            if e.code == 0:
                return {"status": "completed"}
            logger.warning(f"Finalization exited with code {e.code}")
            return {"status": "partial", "exit_code": e.code}
    except Exception as e:
        logger.warning(f"Failed to finalize hypotheses (continuing anyway): {e}")
        return {"status": "failed", "error": str(e)}


def get_hound_results(project_name: str) -> Dict[str, Any]:
    """
    Read Hound's hypotheses.json and return findings.
    
    Returns dict with findings, coverage, and session info.
    """
    project_dir = Path.home() / ".hound" / "projects" / project_name
    hypotheses_file = project_dir / "hypotheses.json"
    
    if not hypotheses_file.exists():
        logger.warning(f"Hypotheses file not found: {hypotheses_file}")
        return {
            "hypotheses": {},
            "coverage": {},
            "session": None
        }
    
    try:
        with open(hypotheses_file, "r") as f:
            hound_data = json.load(f)
        
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
                except Exception:
                    pass
        
        return {
            "hypotheses": hound_data.get("hypotheses", {}),
            "coverage": coverage_data,
            "session": session_data.get("session_id") if session_data else None,
            "project_dir": str(project_dir)
        }
    except Exception as e:
        logger.error(f"Failed to read Hound results: {e}")
        raise RuntimeError(f"Failed to read Hound results: {str(e)}")

