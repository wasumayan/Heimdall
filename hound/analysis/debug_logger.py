"""
Debug logger for agent LLM interactions.
Captures all prompts and responses in a simple log format for easy debugging.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class DebugLogger:
    """Logs all LLM interactions to a log file for debugging."""
    
    def __init__(self, session_id: str, output_dir: Path | None = None):
        """
        Initialize debug logger.
        
        Args:
            session_id: Unique identifier for this session
            output_dir: Directory to save debug logs (defaults to .hound_debug)
        """
        self.session_id = session_id
        # Prefer caller-provided directory, then env var, then workspace-local fallback
        self.output_dir = output_dir
        if not self.output_dir:
            try:
                import os as _os
                env_dir = _os.environ.get("HOUND_DEBUG_DIR")
                if env_dir:
                    self.output_dir = Path(env_dir)
                else:
                    # Use CWD-local debug dir to avoid sandbox permission issues
                    self.output_dir = Path.cwd() / ".hound_debug"
            except Exception:
                self.output_dir = Path.cwd() / ".hound_debug"
        # Ensure directory exists; if creation fails, fall back to CWD
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.output_dir = Path.cwd() / ".hound_debug"
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # Final fallback: disable file logging by pointing to a non-persistent temp-like path
                self.output_dir = Path(".hound_debug")
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"debug_{session_id}_{timestamp}.log"
        # Per-interaction directory (one prompt/response per file)
        try:
            self.interactions_dir = self.output_dir / "interactions"
            self.interactions_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.interactions_dir = self.output_dir
        
        # Initialize log file
        try:
            self._init_log()
        except Exception:
            # If we cannot write the file, disable logging silently
            self.log_file = None
        
        # Track interaction count
        self.interaction_count = 0
    
    def _init_log(self):
        """Initialize the log file with header."""
        header = f"""
================================================================================
HOUND DEBUG LOG
Session ID: {self.session_id}
Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
================================================================================

"""
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write(header)
    
    def log_interaction(
        self,
        system_prompt: str,
        user_prompt: str,
        response: Any,
        schema: Any | None = None,
        duration: float | None = None,
        error: str | None = None,
        tool_calls: list | None = None,
        profile: str | None = None
    ):
        """
        Log a single LLM interaction.
        
        Args:
            system_prompt: System prompt sent to LLM
            user_prompt: User prompt sent to LLM
            response: Response from LLM (string or parsed object)
            schema: Pydantic schema used for parsing (if any)
            duration: Time taken for the interaction
            error: Error message if interaction failed
            tool_calls: List of tool calls generated
        """
        if not self.log_file:
            return
            
        self.interaction_count += 1
        
        # Format response
        if isinstance(response, str):
            response_str = response
        else:
            try:
                response_str = json.dumps(response, indent=2, default=str)
            except Exception:
                response_str = str(response)
        
        # Build log entry
        log_entry = f"""
--------------------------------------------------------------------------------
INTERACTION #{self.interaction_count}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{f'Duration: {duration:.2f}s' if duration else ''}

SYSTEM PROMPT:
{system_prompt}

USER PROMPT:
{user_prompt}

RESPONSE:
{response_str if not error else f'ERROR: {error}'}

"""
        
        # Add tool calls if present
        if tool_calls:
            log_entry += "TOOL CALLS:\n"
            for call in tool_calls:
                tool_name = call.get('tool_name', 'unknown')
                params = json.dumps(call.get('parameters', {}), indent=2)
                log_entry += f"  - {tool_name}: {params}\n"
            log_entry += "\n"
        
        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

        # Also write a per-interaction JSON file for easy inspection
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            idx = f"{self.interaction_count:04d}"
            fname = self.interactions_dir / f"{idx}_{ts}.json"
            record = {
                'time': datetime.now().isoformat(),
                'session_id': self.session_id,
                'profile': profile,
                'system': system_prompt,
                'user': user_prompt,
                'response': response if isinstance(response, str | int | float | list | dict | type(None)) else str(response),
                'schema': str(schema) if schema is not None else None,
                'duration_seconds': duration,
                'error': error,
                'tool_calls': tool_calls or []
            }
            with open(fname, 'w') as jf:
                json.dump(record, jf, indent=2, default=str)
        except Exception:
            pass
    
    def log_event(self, event_type: str, message: str, details: dict | None = None):
        """
        Log a general event (not an LLM interaction).
        
        Args:
            event_type: Type of event (e.g., "Graph Selection", "Hypothesis Update")
            message: Event message
            details: Optional additional details
        """
        if not self.log_file:
            return
            
        details_str = ""
        if details:
            details_str = f"\nDetails: {json.dumps(details, indent=2, default=str)}"
        
        event_log = f"""
--------------------------------------------------------------------------------
EVENT: {event_type}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Message: {message}{details_str}

"""
        
        with open(self.log_file, 'a') as f:
            f.write(event_log)
    
    def finalize(self, summary: dict | None = None):
        """
        Finalize the debug log with summary statistics.
        
        Args:
            summary: Optional summary statistics to include
        """
        if not self.log_file:
            return None
            
        summary_str = ""
        if summary:
            summary_str = "\nSESSION SUMMARY:\n"
            for key, value in summary.items():
                summary_str += f"  {key.replace('_', ' ').title()}: {value}\n"
        
        footer = f"""
================================================================================
{summary_str}
Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Interactions: {self.interaction_count}
================================================================================
"""
        
        with open(self.log_file, 'a') as f:
            f.write(footer)
        
        return self.log_file
