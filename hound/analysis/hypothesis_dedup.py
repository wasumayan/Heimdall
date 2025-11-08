"""LLM-assisted hypothesis deduplication utilities.

Uses a lightweight model profile to compare a new hypothesis against
existing ones in small batches to avoid duplicates.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from llm.unified_client import UnifiedLLMClient


class _DupResult(BaseModel):
    """Structured response from the dedup model for a batch."""
    duplicates: list[str] = Field(default_factory=list, description="List of existing hypothesis IDs that are semantically duplicates")
    rationale: str | None = None


def _get_lightweight_client(cfg: dict[str, Any], debug_logger=None) -> UnifiedLLMClient | None:
    """Try to create a lightweight client; fall back to scout/agent if missing."""
    try:
        return UnifiedLLMClient(cfg=cfg, profile="lightweight", debug_logger=debug_logger)
    except Exception:
        # Fallbacks to stay resilient in older configs
        for profile in ("scout", "agent"):
            try:
                return UnifiedLLMClient(cfg=cfg, profile=profile, debug_logger=debug_logger)
            except Exception:
                continue
    return None


def check_duplicates_llm(
    *,
    cfg: dict[str, Any],
    new_hypothesis: dict[str, Any],
    existing_batch: Iterable[dict[str, Any]],
    debug_logger=None,
) -> set[str]:
    """Return set of existing hypothesis IDs that are duplicates of the new one.

    Input "new_hypothesis" should include: title, description, vulnerability_type, node_refs (list[str]).
    Each item in "existing_batch" should include: id, title, description, vulnerability_type, node_refs.
    """
    client = _get_lightweight_client(cfg, debug_logger)
    if not client:
        return set()

    # Pre-filter: require meaningful, non-generic node overlap to consider duplicates
    def _normalize_nodes(nodes: list[str] | None) -> set[str]:
        bad = {"", "system", "root", "project", "SystemArchitecture"}
        out = set()
        for n in (nodes or []):
            s = str(n).strip()
            if s and s not in bad:
                out.add(s)
        return out

    cand_nodes = _normalize_nodes(new_hypothesis.get("node_refs"))
    # If candidate has no specific nodes, be conservative: rely on exact-title store dedup only
    if not cand_nodes:
        return set()
    # Keep only existing items with overlapping specific nodes
    _filtered: list[dict[str, Any]] = []
    for h in existing_batch:
        if _normalize_nodes(h.get("node_refs")) & cand_nodes:
            # Require same vulnerability type when available
            try:
                vt_new = (new_hypothesis.get("vulnerability_type") or "").strip().lower()
                vt_old = (h.get("vulnerability_type") or "").strip().lower()
            except Exception:
                vt_new = vt_old = ""
            if not vt_new or not vt_old or vt_new == vt_old:
                _filtered.append(h)
    if not _filtered:
        return set()

    # Minimal, clear prompt for robust judging
    system = (
        "You are a fast, precise deduplication helper for security hypotheses.\n"
        "Two items are duplicates iff they describe essentially the SAME bug:\n"
        "- Same root cause AND same concrete code locus (same contract/function/state)\n"
        "- Minor rewording or synonyms do NOT matter\n"
        "They are NOT duplicates if the root cause, scope, or affected code paths differ.\n"
        "Node guidance:\n"
        "- Consider node_refs as the affected code. Require meaningful overlap in specific nodes (functions/contracts).\n"
        "- Ignore generic placeholders like 'system' or 'root'.\n"
        "- If the new item lacks specific nodes, be conservative: only mark duplicate if the title/desc clearly indicate the exact same function/contract and root cause.\n"
        "Return strict JSON only."
    )

    def _fmt_h(h: dict[str, Any]) -> str:
        nid = ",".join(h.get("node_refs") or [])
        return (
            f"id={h.get('id','unknown')}\n"
            f"title={h.get('title') or h.get('description','')}\n"
            f"type={h.get('vulnerability_type','unknown')}\n"
            f"nodes={nid}\n"
            f"desc={h.get('description','')}\n"
        )

    existing_lines = []
    for i, h in enumerate(_filtered, 1):
        existing_lines.append(f"[{i}]\n" + _fmt_h(h))

    user = (
        "NEW HYPOTHESIS:\n" + _fmt_h(new_hypothesis) + "\n" +
        "EXISTING HYPOTHESES (CANDIDATE DUPLICATES):\n" + "\n".join(existing_lines) + "\n\n" +
        "Task: Identify duplicates by id.\n"
        "Output strict JSON: {\"duplicates\":[\"hyp_abc123\",...],\"rationale\":\"short reason\"}"
    )

    try:
        result = client.parse(system=system, user=user, schema=_DupResult)
        ids = set([str(x).strip() for x in (result.duplicates or []) if str(x).strip()])
        return ids
    except Exception:
        # Never block on dedup failures
        return set()
