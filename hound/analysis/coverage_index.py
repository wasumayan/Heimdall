"""Concurrent coverage index to track what was investigated.

Stores per-node coverage stats and investigation records to help Strategist
avoid redundant work while enabling revisits when new evidence appears.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .concurrent_knowledge import ConcurrentFileStore


@dataclass
class InvestigationRecord:
    frame_id: str | None
    node_ids: list[str]
    status: str  # planned|in_progress|done|dropped|superseded
    timestamp: str = datetime.now().isoformat()


class CoverageIndex(ConcurrentFileStore):
    """Per-project coverage index with atomic updates."""

    def _get_empty_data(self) -> dict:
        return {
            "nodes": {},  # node_id -> {last_seen, seen_count, evidence_count}
            "cards": {},  # card_id -> {last_seen, seen_count}
            "investigations": [],
            "metadata": {"last_modified": datetime.now().isoformat()},
        }

    def touch_node(self, node_id: str) -> None:
        def update(data):
            nodes = data.setdefault("nodes", {})
            rec = nodes.get(node_id, {"last_seen": None, "seen_count": 0, "evidence_count": 0})
            rec["last_seen"] = datetime.now().isoformat()
            rec["seen_count"] = int(rec.get("seen_count", 0)) + 1
            nodes[node_id] = rec
            data["metadata"]["last_modified"] = datetime.now().isoformat()
            return data, True

        self.update_atomic(update)

    def add_evidence(self, node_id: str) -> None:
        def update(data):
            nodes = data.setdefault("nodes", {})
            rec = nodes.get(node_id, {"last_seen": None, "seen_count": 0, "evidence_count": 0})
            rec["evidence_count"] = int(rec.get("evidence_count", 0)) + 1
            rec["last_seen"] = datetime.now().isoformat()
            nodes[node_id] = rec
            data["metadata"]["last_modified"] = datetime.now().isoformat()
            return data, True

        self.update_atomic(update)

    def touch_card(self, card_id: str) -> None:
        def update(data):
            cards = data.setdefault("cards", {})
            rec = cards.get(card_id, {"last_seen": None, "seen_count": 0})
            rec["last_seen"] = datetime.now().isoformat()
            rec["seen_count"] = int(rec.get("seen_count", 0)) + 1
            cards[card_id] = rec
            data["metadata"]["last_modified"] = datetime.now().isoformat()
            return data, True
        self.update_atomic(update)

    def record_investigation(self, frame_id: str | None, node_ids: list[str], status: str) -> None:
        def update(data):
            inv = asdict(InvestigationRecord(frame_id=frame_id, node_ids=node_ids, status=status))
            data.setdefault("investigations", []).append(inv)
            data["metadata"]["last_modified"] = datetime.now().isoformat()
            return data, True

        self.update_atomic(update)

    def summarize(self, limit: int = 100) -> str:
        """Return a compact text summary for planner prompts."""
        lock = self._acquire_lock()
        try:
            data = self._load_data()
            nodes = data.get("nodes", {})
            # Sort by recency
            items = []
            for nid, rec in nodes.items():
                items.append((nid, rec.get("last_seen") or "", rec.get("seen_count", 0), rec.get("evidence_count", 0)))
            items.sort(key=lambda x: x[1], reverse=True)
            lines = []
            for nid, last, seen, evid in items[:limit]:
                lines.append(f"{nid}: seen={seen}, evidence={evid}, last={last}")
            return "\n".join(lines)
        finally:
            self._release_lock(lock)

    def compute_stats(self, graphs_dir: Path, manifest_dir: Path) -> dict[str, Any]:
        """Compute total and visited counts for nodes and cards with percentages."""
        lock = self._acquire_lock()
        try:
            data = self._load_data()
        finally:
            self._release_lock(lock)

        # Collect all nodes from all graph_*.json
        total_nodes: set[str] = set()
        try:
            for gfile in Path(graphs_dir).glob('graph_*.json'):
                import json as _json
                try:
                    gd = _json.loads(Path(gfile).read_text())
                    for n in gd.get('nodes', []) or []:
                        nid = n.get('id')
                        if nid:
                            total_nodes.add(str(nid))
                except Exception:
                    continue
        except Exception:
            pass

        visited_nodes = set((data.get('nodes') or {}).keys())

        # Collect total cards
        total_cards: set[str] = set()
        card_store = Path(graphs_dir) / 'card_store.json'
        if card_store.exists():
            try:
                import json as _json
                store = _json.loads(card_store.read_text())
                if isinstance(store, dict):
                    total_cards.update([str(k) for k in store.keys()])
            except Exception:
                pass
        else:
            cards_jsonl = Path(manifest_dir) / 'cards.jsonl'
            if cards_jsonl.exists():
                try:
                    for line in cards_jsonl.read_text().splitlines():
                        try:
                            import json as _json
                            cd = _json.loads(line)
                            cid = cd.get('id')
                            if cid:
                                total_cards.add(str(cid))
                        except Exception:
                            continue
                except Exception:
                    pass

        visited_cards = set((data.get('cards') or {}).keys())

        def pct(x: int, y: int) -> float:
            return round((x / y * 100.0) if y else 0.0, 2)

        stats = {
            'nodes': {
                'total': len(total_nodes),
                'visited': len(visited_nodes & total_nodes) if total_nodes else len(visited_nodes),
                'percent': pct(len(visited_nodes & total_nodes) if total_nodes else len(visited_nodes), len(total_nodes)),
            },
            'cards': {
                'total': len(total_cards),
                'visited': len(visited_cards & total_cards) if total_cards else len(visited_cards),
                'percent': pct(len(visited_cards & total_cards) if total_cards else len(visited_cards), len(total_cards)),
            }
        }
        return stats
