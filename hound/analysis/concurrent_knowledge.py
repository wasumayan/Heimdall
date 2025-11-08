"""
Concurrent knowledge management system for multi-process graph and hypothesis operations.

This module provides thread-safe, file-based storage for knowledge graphs and 
vulnerability hypotheses, allowing multiple agents to collaborate on analysis.
"""

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import portalocker

# ============================================================================
# Base Concurrent Store
# ============================================================================

class ConcurrentFileStore(ABC):
    """Base class for file-based storage with process-safe locking."""
    
    def __init__(self, file_path: Path, agent_id: str | None = None):
        self.file_path = Path(file_path)
        self.agent_id = agent_id or "anonymous"
        self.lock_path = self.file_path.with_suffix('.lock')
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize or fetch a per-file thread lock to guard against races within a single process
        self._thread_lock = self._get_thread_lock(self.file_path)
        
        if not self.file_path.exists():
            self._save_data(self._get_empty_data())
    
    @abstractmethod
    def _get_empty_data(self) -> dict:
        """Return initial empty data structure."""
        pass
    
    def _acquire_lock(self, timeout: float = 10.0) -> Any:
        """Acquire exclusive lock on storage file."""
        start_time = time.time()
        lock_file = open(self.lock_path, 'w')
        
        while True:
            try:
                portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                return lock_file
            except (OSError, portalocker.exceptions.LockException) as exc:
                # Retry until timeout to handle concurrent access across processes
                if time.time() - start_time > timeout:
                    lock_file.close()
                    raise TimeoutError(f"Lock timeout: {self.file_path}") from exc
                time.sleep(0.05)
    
    def _release_lock(self, lock_file: Any):
        """Release file lock."""
        try:
            portalocker.unlock(lock_file)
            lock_file.close()
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass
    
    def _load_data(self) -> dict:
        """Load data from file."""
        try:
            with open(self.file_path) as f:
                return json.load(f)
        except Exception:
            return self._get_empty_data()
    
    def _save_data(self, data: dict):
        """Save data atomically using a unique temp file to avoid races."""
        import tempfile
        # Create temp file in same directory for atomic replace on same filesystem
        with tempfile.NamedTemporaryFile('w', dir=str(self.file_path.parent), prefix=self.file_path.stem + '.', suffix='.tmp', delete=False) as tf:
            json.dump(data, tf, indent=2, default=str)
            tmp_name = Path(tf.name)
        tmp_name.replace(self.file_path)
    
    def update_atomic(self, update_func) -> Any:
        """Atomically read, update, and write data."""
        # Ensure only one thread in this process performs the critical section at a time
        with self._thread_lock:
            lock = self._acquire_lock()
            try:
                data = self._load_data()
                updated_data, result = update_func(data)
                if updated_data is not None:
                    self._save_data(updated_data)
                return result
            finally:
                self._release_lock(lock)

    # ---------- Internal: per-file thread lock registry ----------
    _locks_registry: dict[str, threading.RLock] = {}
    _locks_registry_guard = threading.Lock()

    @classmethod
    def _get_thread_lock(cls, file_path: Path) -> threading.RLock:
        """Return a re-entrant lock keyed by absolute file path string."""
        key = str(Path(file_path).resolve())
        with cls._locks_registry_guard:
            lk = cls._locks_registry.get(key)
            if lk is None:
                lk = threading.RLock()
                cls._locks_registry[key] = lk
            return lk


# ============================================================================
# Hypothesis Data Structures  
# ============================================================================

class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    INVESTIGATING = "investigating"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class Evidence:
    """Evidence for/against a hypothesis."""
    description: str
    type: str  # supports/refutes/related
    confidence: float = 0.7
    node_refs: list[str] = field(default_factory=list)
    created_by: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Hypothesis:
    """Vulnerability hypothesis."""
    title: str
    description: str
    vulnerability_type: str
    severity: str  # low/medium/high/critical
    confidence: float = 0.5
    status: str = "proposed"
    node_refs: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    reasoning: str = ""
    properties: dict[str, Any] = field(default_factory=dict)  # Store graph name, etc.
    created_by: str | None = None
    reported_by_model: str | None = None  # Legacy field for backward compatibility
    junior_model: str | None = None  # The agent model that discovered the vulnerability
    senior_model: str | None = None  # The guidance/deep think model that analyzed it
    # Session tagging
    session_id: str | None = None
    visibility: str = "global"  # 'global' or 'session'
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = ""
    
    def __post_init__(self):
        if not self.id:
            content = f"{self.title}{self.vulnerability_type}{''.join(self.node_refs)}"
            self.id = f"hyp_{hashlib.md5(content.encode()).hexdigest()[:12]}"


# ============================================================================
# Hypothesis Store
# ============================================================================

class HypothesisStore(ConcurrentFileStore):
    """Manages vulnerability hypotheses with concurrent access."""
    
    def _get_empty_data(self) -> dict:
        return {
            "version": "1.0",
            "hypotheses": {},
            "metadata": {
                "total": 0,
                "confirmed": 0,
                "last_modified": datetime.now().isoformat()
            }
        }
    
    def propose(self, hypothesis: Hypothesis) -> tuple[bool, str]:
        """Propose a new hypothesis with improved duplicate detection."""
        def update(data):
            hypotheses = data["hypotheses"]
            
            # Duplicate check: keep conservative to allow multiple issues on same nodes
            for h_id, h in hypotheses.items():
                # Exact title match (case-insensitive)
                if (h.get("title", "").lower() or "") == hypothesis.title.lower():
                    return data, (False, f"Duplicate title: {h_id}")
            
            hypothesis.created_by = self.agent_id
            hypotheses[hypothesis.id] = asdict(hypothesis)
            data["metadata"]["total"] = len(hypotheses)
            data["metadata"]["last_modified"] = datetime.now().isoformat()
            
            return data, (True, hypothesis.id)
        
        return self.update_atomic(update)

    def list_all(self) -> list[dict]:
        """Return all hypotheses as a list (non-mutating)."""
        lock = self._acquire_lock()
        try:
            data = self._load_data()
            hyps = data.get("hypotheses", {}) or {}
            # Ensure each has an id field populated (older data safety)
            out: list[dict] = []
            for h in hyps.values():
                if "id" not in h:
                    try:
                        # Best-effort backfill; mirrors Hypothesis.__post_init__
                        content = f"{h.get('title','')}{h.get('vulnerability_type','')}{''.join(h.get('node_refs') or [])}"
                        import hashlib as _hashlib
                        h["id"] = f"hyp_{_hashlib.md5(content.encode()).hexdigest()[:12]}"
                    except Exception:
                        pass
                out.append(h)
            return out
        finally:
            self._release_lock(lock)
    
    def add_evidence(self, hypothesis_id: str, evidence: Evidence) -> bool:
        """Add evidence to a hypothesis."""
        def update(data):
            if hypothesis_id not in data["hypotheses"]:
                return data, False
            
            hyp = data["hypotheses"][hypothesis_id]
            evidence.created_by = self.agent_id
            hyp["evidence"].append(asdict(evidence))
            
            # Auto-adjust status
            supporting = sum(1 for e in hyp["evidence"] if e["type"] == "supports")
            refuting = sum(1 for e in hyp["evidence"] if e["type"] == "refutes")
            
            if refuting > supporting * 2:
                hyp["status"] = "refuted"
            elif supporting > 3:
                hyp["status"] = "supported"
            elif supporting > 0:
                hyp["status"] = "investigating"
            
            return data, True
        
        return self.update_atomic(update)
    
    def adjust_confidence(self, hypothesis_id: str, confidence: float, reason: str) -> bool:
        """Adjust hypothesis confidence and optionally add QA comment."""
        def update(data):
            if hypothesis_id not in data["hypotheses"]:
                return data, False
            
            hyp = data["hypotheses"][hypothesis_id]
            hyp["confidence"] = confidence
            
            # Store QA comment/reasoning if provided (backwards compatible)
            if reason:
                hyp["qa_comment"] = reason
            
            # Auto-update status (analysis agent can only reject, not confirm)
            # Only the finalize agent can set status to "confirmed"
            if confidence <= 0.1:
                hyp["status"] = "rejected"
            
            return data, True
        
        return self.update_atomic(update)
    
    def get_by_node(self, node_id: str) -> list[dict]:
        """Get hypotheses for a node."""
        lock = self._acquire_lock()
        try:
            data = self._load_data()
            return [h for h in data["hypotheses"].values() if node_id in h.get("node_refs", [])]
        finally:
            self._release_lock(lock)


# ============================================================================
# Graph Store
# ============================================================================

class GraphStore(ConcurrentFileStore):
    """Manages graph files with concurrent access."""
    
    def _get_empty_data(self) -> dict:
        return {
            "name": self.file_path.stem,
            "created_at": datetime.now().isoformat(),
            "nodes": [],
            "edges": [],
            "metadata": {"version": "1.0"}
        }
    
    def save_graph(self, graph_data: dict) -> bool:
        """Save entire graph data atomically."""
        def update(data):
            # Replace entire graph data
            return graph_data, True
        
        return self.update_atomic(update)
    
    def load_graph(self) -> dict:
        """Load graph data with shared lock."""
        lock = self._acquire_lock()
        try:
            return self._load_data()
        finally:
            self._release_lock(lock)
    
    def update_nodes(self, node_updates: list[dict]) -> bool:
        """Update specific nodes in the graph."""
        def update(data):
            # Create a map of node IDs to updates
            update_map = {update['id']: update for update in node_updates}
            
            # Update existing nodes
            for i, node in enumerate(data.get('nodes', [])):
                if node['id'] in update_map:
                    # Merge updates into existing node
                    node.update(update_map[node['id']])
            
            return data, True
        
        return self.update_atomic(update)
