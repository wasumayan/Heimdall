"""Session manager for audit runs.

Provides simple session directory management under a project directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SessionInfo:
    session_id: str
    path: Path


class SessionManager:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.sessions_dir = self.project_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create(self, session_id: str | None = None) -> SessionInfo:
        sid = session_id or f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sp = self.sessions_dir / sid
        sp.mkdir(parents=True, exist_ok=True)
        return SessionInfo(session_id=sid, path=sp)

    def get(self, session_id: str) -> SessionInfo | None:
        sp = self.sessions_dir / session_id
        if sp.exists():
            return SessionInfo(session_id=session_id, path=sp)
        return None

    def get_or_create(self, session_id: str | None = None, new_session: bool = False) -> SessionInfo:
        if session_id and not new_session:
            found = self.get(session_id)
            if found:
                return found
        return self.create(session_id)

