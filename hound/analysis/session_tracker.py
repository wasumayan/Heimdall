"""Session tracker with coverage tracking for audit sessions."""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SessionCoverage:
    """Track coverage statistics for a session."""
    visited_nodes: set[str] = field(default_factory=set)
    visited_cards: set[str] = field(default_factory=set)
    total_nodes: int = 0
    total_cards: int = 0
    # Per-node/card visit counts
    node_visit_counts: dict[str, int] = field(default_factory=dict)
    card_visit_counts: dict[str, int] = field(default_factory=dict)
    # Known IDs to bound coverage
    known_node_ids: set[str] = field(default_factory=set)
    known_card_ids: set[str] = field(default_factory=set)
    
    def add_node(self, node_id: str):
        """Mark a node as visited."""
        self.visited_nodes.add(node_id)
        self.node_visit_counts[node_id] = int(self.node_visit_counts.get(node_id, 0)) + 1
    
    def add_card(self, card_id: str):
        """Mark a card as visited."""
        self.visited_cards.add(card_id)
        self.card_visit_counts[card_id] = int(self.card_visit_counts.get(card_id, 0)) + 1
    
    def get_stats(self) -> dict[str, Any]:
        """Get coverage statistics bounded to known IDs to avoid >100%."""
        # Default totals
        nodes_total = len(self.known_node_ids) if self.known_node_ids else self.total_nodes
        cards_total = len(self.known_card_ids) if self.known_card_ids else self.total_cards
        # Bound visited to known sets when available
        nodes_visited = len(self.visited_nodes & self.known_node_ids) if self.known_node_ids else len(self.visited_nodes)
        cards_visited = len(self.visited_cards & self.known_card_ids) if self.known_card_ids else len(self.visited_cards)
        
        def pct(a: int, b: int) -> float:
            return round((a / b * 100.0) if b else 0.0, 1)
        
        return {
            'nodes': {
                'visited': nodes_visited,
                'total': nodes_total,
                'percent': pct(nodes_visited, nodes_total)
            },
            'cards': {
                'visited': cards_visited,
                'total': cards_total,
                'percent': pct(cards_visited, cards_total)
            },
            'visited_node_ids': list(self.visited_nodes),
            'visited_card_ids': list(self.visited_cards),
            'node_visit_counts': dict(self.node_visit_counts)
        }


class SessionTracker:
    """Track an audit session including coverage, investigations, and planning."""
    
    def __init__(self, session_dir: Path, session_id: str):
        """Initialize session tracker.
        
        Args:
            session_dir: Directory to store session data
            session_id: Unique session identifier
        """
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.session_file = self.session_dir / f"{session_id}.json"
        self.lock = threading.Lock()
        
        # Initialize or load session data
        self.session_data = self._load_or_init()
        
        # Initialize coverage tracker
        self.coverage = SessionCoverage()
        if 'coverage' in self.session_data:
            cov_data = self.session_data['coverage']
            self.coverage.visited_nodes = set(cov_data.get('visited_node_ids', []))
            self.coverage.visited_cards = set(cov_data.get('visited_card_ids', []))
            self.coverage.total_nodes = cov_data.get('nodes', {}).get('total', 0)
            self.coverage.total_cards = cov_data.get('cards', {}).get('total', 0)
    
    def _load_or_init(self) -> dict[str, Any]:
        """Load existing session or initialize new one."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        if self.session_file.exists():
            try:
                with open(self.session_file) as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Initialize new session
        return {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'status': 'active',
            'models': {},
            'investigations': [],
            'planning_history': [],
            'token_usage': {},
            'coverage': {}
        }
    
    def set_models(self, scout_model: str, strategist_model: str):
        """Set the models being used."""
        self.session_data['models'] = {
            'scout': scout_model,
            'strategist': strategist_model
        }
        self._save()
    
    def initialize_coverage(self, graphs_dir: Path, manifest_dir: Path):
        """Initialize coverage tracking by counting total nodes and cards.
        
        Args:
            graphs_dir: Directory containing graph files
            manifest_dir: Directory containing manifest files
        """
        # Count nodes from graphs and record known IDs
        total_nodes = 0
        if graphs_dir.exists():
            for graph_file in graphs_dir.glob("graph_*.json"):
                try:
                    with open(graph_file) as f:
                        graph_data = json.load(f)
                        nodes = graph_data.get('nodes', [])
                        total_nodes += len(nodes)
                        for n in nodes or []:
                            nid = n.get('id')
                            if nid is not None:
                                self.coverage.known_node_ids.add(str(nid))
                except Exception:
                    pass
        
        # Count cards from manifest
        total_cards = 0
        manifest_file = manifest_dir / "manifest.json" if manifest_dir.exists() else None
        if manifest_file and manifest_file.exists():
            try:
                with open(manifest_file) as f:
                    manifest_data = json.load(f)
                    # Try both formats - num_cards or files array
                    if 'num_cards' in manifest_data:
                        total_cards = manifest_data['num_cards']
                    elif 'files' in manifest_data:
                        total_cards = len(manifest_data['files'])
            except Exception:
                pass
        
        # Build known card IDs and file->cards mapping for accurate tracking
        try:
            self._file_to_cards: dict[str, list[str]] = {}
            cards_jsonl = manifest_dir / 'cards.jsonl'
            if cards_jsonl.exists():
                with open(cards_jsonl) as f:
                    for line in f:
                        try:
                            card = json.loads(line)
                            cid = card.get('id')
                            rel = card.get('relpath')
                            if cid:
                                self.coverage.known_card_ids.add(str(cid))
                            if rel and cid:
                                self._file_to_cards.setdefault(rel, []).append(str(cid))
                        except Exception:
                            continue
            files_json = manifest_dir / 'files.json'
            if files_json.exists():
                with open(files_json) as f:
                    files_list = json.load(f)
                if isinstance(files_list, list):
                    for fi in files_list:
                        rel = fi.get('relpath')
                        cids = fi.get('card_ids', []) or []
                        if rel and cids:
                            self._file_to_cards[rel] = [str(x) for x in cids]
                            for x in cids:
                                self.coverage.known_card_ids.add(str(x))
        except Exception:
            pass

        self.coverage.total_nodes = total_nodes
        self.coverage.total_cards = total_cards
        self._save()
    
    def track_node_visit(self, node_id: str):
        """Track that a node was visited during investigation."""
        with self.lock:
            self.coverage.add_node(node_id)
            self._save()
    
    def track_card_visit(self, card_path: str):
        """Track that a code card was analyzed."""
        with self.lock:
            # Try to map file path to card IDs for accurate card coverage
            ids: list[str] = []
            try:
                # Normalize path to relpath if possible
                rel = card_path
                # If the provided path is absolute or has prefixes, try to use as-is in mapping first
                if hasattr(self, '_file_to_cards'):
                    if rel in self._file_to_cards:
                        ids = list(self._file_to_cards.get(rel, []))
                    else:
                        # Try stripping leading slashes
                        rel2 = rel.lstrip('/')
                        ids = list(self._file_to_cards.get(rel2, []))
            except Exception:
                ids = []
            if ids:
                for cid in ids:
                    self.coverage.add_card(cid)
            else:
                # Fall back to recording the path (won't count toward known cards)
                self.coverage.add_card(card_path)
            self._save()
    
    def track_nodes_batch(self, node_ids: list[str]):
        """Track multiple nodes visited at once."""
        with self.lock:
            for node_id in node_ids:
                self.coverage.add_node(node_id)
            self._save()
    
    def track_cards_batch(self, card_paths: list[str]):
        """Track multiple cards analyzed at once."""
        with self.lock:
            for card_path in card_paths:
                self.coverage.add_card(card_path)
            self._save()
    
    def add_investigation(self, investigation: dict[str, Any]):
        """Add an investigation to the session history."""
        with self.lock:
            self.session_data['investigations'].append({
                'timestamp': datetime.now().isoformat(),
                **investigation
            })
            self._save()
    
    def add_planning(self, plan_items: list[dict[str, Any]]):
        """Add a planning batch to the history."""
        with self.lock:
            self.session_data['planning_history'].append({
                'timestamp': datetime.now().isoformat(),
                'items': plan_items
            })
            self._save()
    
    def update_token_usage(self, tokens: dict[str, Any]):
        """Update token usage statistics."""
        with self.lock:
            # The token tracker passes a complex structure with total_usage, by_model, and history
            # We'll store the entire structure
            self.session_data['token_usage'] = tokens
            self._save()
    
    def get_coverage_stats(self) -> dict[str, Any]:
        """Get current coverage statistics."""
        return self.coverage.get_stats()
    
    def finalize(self, status: str = 'completed'):
        """Mark session as finalized."""
        with self.lock:
            self.session_data['status'] = status
            self.session_data['end_time'] = datetime.now().isoformat()
            self.session_data['coverage'] = self.coverage.get_stats()
            self._save()
    
    def set_status(self, status: str):
        """Set current session status without finalizing."""
        with self.lock:
            self.session_data['status'] = status
            # Do not touch end_time here; only set on finalize
            self._save()
    
    def _save(self):
        """Save session data to file (call within lock)."""
        try:
            # Include current coverage in saved data
            self.session_data['coverage'] = self.coverage.get_stats()
            
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save session data: {e}")
