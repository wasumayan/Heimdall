"""Run tracking for agent analysis."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


class RunTracker:
    """Tracks agent run metadata and updates in real-time."""
    
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.start_time = time.time()
        self.data = {
            'run_id': None,
            'command_args': [],
            'session_id': None,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'runtime_seconds': 0,
            'status': 'running',
            'token_usage': {},
            'investigations': [],
            'errors': []
        }
        self._lock = Lock()
        self._save()
    
    def set_run_info(self, run_id: str, command_args: list[str]):
        """Set basic run information."""
        with self._lock:
            self.data['run_id'] = run_id
            self.data['command_args'] = command_args
            self._save()

    def set_session_id(self, session_id: str):
        """Associate a session ID with this run."""
        with self._lock:
            self.data['session_id'] = session_id
            self._update_runtime()
            self._save()
    
    def update_token_usage(self, token_summary: dict[str, Any]):
        """Update token usage statistics."""
        with self._lock:
            # Exclude the detailed history to keep the file size manageable
            filtered_summary = {
                'total_usage': token_summary.get('total_usage', {}),
                'by_model': token_summary.get('by_model', {})
            }
            self.data['token_usage'] = filtered_summary
            self._update_runtime()
            self._save()
    
    def add_investigation(self, investigation: dict[str, Any]):
        """Add an investigation result."""
        with self._lock:
            self.data['investigations'].append(investigation)
            self._update_runtime()
            self._save()
    
    def add_error(self, error: str):
        """Add an error message."""
        with self._lock:
            self.data['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error
            })
            self._update_runtime()
            self._save()
    
    def finalize(self, status: str = 'completed'):
        """Finalize the run."""
        with self._lock:
            self.data['status'] = status
            self.data['end_time'] = datetime.now().isoformat()
            self._update_runtime()
            self._save()
    
    def _update_runtime(self):
        """Update the runtime in seconds."""
        self.data['runtime_seconds'] = round(time.time() - self.start_time, 2)
    
    def _save(self):
        """Save current state to file (called within lock)."""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save run tracker: {e}", file=sys.stderr)
