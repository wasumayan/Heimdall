"""Concurrent project-wide plan ledger (read-mostly).

Tracks normalized plan frames across sessions for transparency and optional
de-duplication. It does NOT block repeated frames — different sessions/models
can analyze the same items intentionally.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .concurrent_knowledge import ConcurrentFileStore


def _norm_key(question: str, artifact_refs: list[str]) -> str:
    q = (question or '').strip()
    arts = ','.join(sorted([a.strip() for a in (artifact_refs or [])]))
    h = hashlib.md5(f"{q}|{arts}".encode()).hexdigest()[:12]
    return f"frame_{h}"


@dataclass
class LedgerEntry:
    key: str
    question: str
    artifact_refs: list[str] = field(default_factory=list)
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    sessions: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)  # e.g., strategist model signatures
    count: int = 0


class PlanLedger(ConcurrentFileStore):
    def _get_empty_data(self) -> dict:
        return {"frames": {}, "metadata": {"last_modified": datetime.now().isoformat()}}

    def record(self, session_id: str, question: str, artifact_refs: list[str], model_sig: str | None = None) -> str:
        key = _norm_key(question, artifact_refs)

        def update(data):
            frames = data.setdefault('frames', {})
            if key not in frames:
                entry = asdict(LedgerEntry(key=key, question=question, artifact_refs=artifact_refs))
                entry['sessions'] = [session_id]
                if model_sig:
                    entry['models'] = [model_sig]
                entry['count'] = 1
                frames[key] = entry
            else:
                e = frames[key]
                e['last_seen'] = datetime.now().isoformat()
                e['count'] = int(e.get('count', 0)) + 1
                if session_id and session_id not in e.get('sessions', []):
                    e.setdefault('sessions', []).append(session_id)
                if model_sig and model_sig not in e.get('models', []):
                    e.setdefault('models', []).append(model_sig)
            data['metadata']['last_modified'] = datetime.now().isoformat()
            return data, key

        return self.update_atomic(update)

    def recent(self, k: int = 10) -> list[dict[str, Any]]:
        lock = self._acquire_lock()
        try:
            data = self._load_data()
            frames = list(data.get('frames', {}).values())
            frames.sort(key=lambda e: e.get('last_seen', ''), reverse=True)
            return frames[:k]
        finally:
            self._release_lock(lock)

    def summarize_recent(self, k: int = 10) -> str:
        items = self.recent(k)
        lines = []
        for e in items:
            lines.append(f"{e.get('question','')} | refs={','.join(e.get('artifact_refs',[]))} | seen={e.get('count',0)}")
        return '\n'.join(lines)

