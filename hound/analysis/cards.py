"""Card index and content extraction utilities shared by agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_card_index(graphs_metadata_path: Path, manifest_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Load a card index and file→card mapping.

    Returns (card_index, file_to_cards).
    """
    card_index: dict[str, dict[str, Any]] = {}
    file_to_cards: dict[str, list[str]] = {}

    # Prefer graph card store if available
    try:
        graphs_dir = Path(graphs_metadata_path).parent
        card_store = graphs_dir / 'card_store.json'
        if card_store.exists():
            with open(card_store) as f:
                store = json.load(f)
                if isinstance(store, dict):
                    for cid, card in store.items():
                        if cid and isinstance(card, dict):
                            card_index[cid] = card
    except Exception:
        pass

    # Also load manifest cards
    manifest_file = Path(manifest_path) / "cards.jsonl"
    if manifest_file.exists():
        with open(manifest_file) as f:
            for line in f:
                try:
                    card = json.loads(line)
                    cid = card.get('id')
                    if cid and cid not in card_index:
                        card_index[cid] = card
                        rel = card.get('relpath')
                        if rel:
                            file_to_cards.setdefault(rel, []).append(cid)
                except json.JSONDecodeError:
                    continue

    # Load files.json to get ordered mapping relpath -> card_ids
    files_json = Path(manifest_path) / 'files.json'
    if files_json.exists():
        try:
            with open(files_json) as f:
                files_list = json.load(f)
            if isinstance(files_list, list):
                for fi in files_list:
                    rel = fi.get('relpath')
                    cids = fi.get('card_ids', []) or []
                    if rel and isinstance(cids, list):
                        file_to_cards[rel] = cids
        except Exception:
            pass

    return card_index, file_to_cards


def extract_card_content(card: dict[str, Any], repo_root: Path | None) -> str:
    """Get best-available content from a card record."""
    content = card.get('content')
    if content:
        return content
    # Try reconstructing from source if we know offsets
    try:
        rel = card.get('relpath')
        cs = card.get('char_start')
        ce = card.get('char_end')
        if repo_root and rel and isinstance(cs, int) and isinstance(ce, int) and ce > cs:
            fpath = repo_root / rel
            if fpath.exists():
                text = fpath.read_text(encoding='utf-8', errors='ignore')
                return text[cs:ce]
    except Exception:
        pass
    # Fallback to peek head/tail if content missing
    head = card.get('peek_head', '') or ''
    tail = card.get('peek_tail', '') or ''
    return (head + ("\n" if head and tail else "") + tail).strip()

