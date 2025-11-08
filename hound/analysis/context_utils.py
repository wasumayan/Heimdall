"""Shared context helpers for Strategist and Scout.

Phase 2 extracts reusable context formatting and summarization utilities to
keep agent classes lean and avoid large files.
"""

from __future__ import annotations

from typing import Any


def format_graph_for_display(graph_data: dict[str, Any], graph_name: str, max_edges: int = 50) -> list[str]:
    """Format a graph for compact display with observations/assumptions.

    Mirrors the display used by the junior agent so Strategist and Scout share
    a consistent view.
    """
    lines: list[str] = []
    lines.append(f"\n--- Graph: {graph_name} ---")

    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    lines.append(f"Total: {len(nodes)} nodes, {len(edges)} edges")
    lines.append("USE EXACT NODE IDS AS SHOWN BELOW - NO VARIATIONS!\n")

    if nodes:
        lines.append("AVAILABLE NODES (use these EXACT IDs with load_nodes):")
        lines.append("[S]=small (1-2 cards), [M]=medium (3-5), [L]=large (6+)")
        for node in nodes:
            node_id = node.get('id', 'unknown')
            node_label = node.get('label', node_id)
            node_type = node.get('type', 'unknown')

            source_refs = node.get('source_refs', []) or []
            card_count = len(source_refs)
            if card_count == 0:
                size_indicator = "[∅]"
            elif card_count <= 2:
                size_indicator = "[S]"
            elif card_count <= 5:
                size_indicator = "[M]"
            else:
                size_indicator = f"[L:{card_count}]"

            lines.append(f"  {size_indicator} [{node_id}] → {node_label} ({node_type})")

            observations = node.get('observations', [])
            assumptions = node.get('assumptions', [])
            annots: list[str] = []

            if observations:
                # take up to 2
                obs_strs = []
                for obs in observations[:2]:
                    if isinstance(obs, dict):
                        desc = obs.get('description', obs.get('content', str(obs)))
                        obs_strs.append(desc)
                    else:
                        obs_strs.append(str(obs))
                if obs_strs:
                    annots.append(f"obs:{'; '.join(obs_strs)}")

            if assumptions:
                assum_strs = []
                for assum in assumptions[:2]:
                    if isinstance(assum, dict):
                        desc = assum.get('description', assum.get('content', str(assum)))
                        assum_strs.append(desc)
                    else:
                        assum_strs.append(str(assum))
                if assum_strs:
                    annots.append(f"asm:{'; '.join(assum_strs)}")

            if annots:
                lines.append(f"    [{' | '.join(annots)}]")

    if edges:
        lines.append("\nEDGE TYPES:")
        edge_types: dict[str, int] = {}
        for edge in edges:
            t = edge.get('type', 'unknown')
            edge_types[t] = edge_types.get(t, 0) + 1
        for edge_type, count in edge_types.items():
            lines.append(f"  • {edge_type}: {count} edges")

        lines.append("\nEDGES (compact):")
        for edge in edges[:max_edges]:
            src = edge.get('source_id') or edge.get('source') or edge.get('src')
            dst = edge.get('target_id') or edge.get('target') or edge.get('dst')
            etype = edge.get('type', 'rel')
            edge_line = f"  {etype} {src}->{dst}"

            edge_obs = edge.get('observations', [])
            edge_assum = edge.get('assumptions', [])
            edge_annots: list[str] = []
            if edge_obs:
                edge_annots.append(f"obs:{'; '.join(str(o) for o in edge_obs[:2])}")
            if edge_assum:
                edge_annots.append(f"asm:{'; '.join(str(a) for a in edge_assum[:2])}")
            if edge_annots:
                edge_line += f" [{' | '.join(edge_annots)}]"
            lines.append(edge_line)

        if len(edges) > max_edges:
            lines.append(f"  ... and {len(edges) - max_edges} more edges")

    return lines


def hypotheses_summary(hypothesis_data: dict[str, Any], limit: int = 10) -> str:
    """Return a compact textual summary of top hypotheses for prompts."""
    hyps = hypothesis_data.get("hypotheses", {})
    if not hyps:
        return ""
    # Sort by confidence desc
    items = sorted(hyps.items(), key=lambda kv: kv[1].get('confidence', 0), reverse=True)[:limit]
    lines = []
    for _, h in items:
        desc = h.get('description', h.get('title', 'Unknown'))
        conf = h.get('confidence', 0)
        lines.append(f"- {desc} (confidence: {conf:.0%})")
    return "\n".join(lines)


def build_investigation_context(
    investigation_goal: str,
    available_graphs: dict[str, Any],
    loaded_data: dict[str, Any],
    memory_notes: list[str] | None = None,
    action_log: list[dict[str, Any]] | None = None,
) -> str:
    """Build a consistent context string for LLM prompts from core fields.

    This mirrors the structure used by the junior agent for familiarity.
    """
    memory_notes = memory_notes or []
    action_log = action_log or []

    parts: list[str] = []
    parts.append("=== INVESTIGATION GOAL ===")
    parts.append(investigation_goal)
    parts.append("")

    # Available graphs
    parts.append("=== AVAILABLE GRAPHS ===")
    parts.append("Use EXACT graph names as shown below:")
    # Handle case where the key exists but value is None
    sys_g_name = ((loaded_data.get('system_graph') or {}).get('name')
                  if loaded_data else None)
    for name in available_graphs.keys():
        if sys_g_name and name == sys_g_name:
            parts.append(f"• {name} [SYSTEM - AUTO-LOADED, see nodes below]")
        else:
            parts.append(f"• {name}")
    parts.append("")

    if memory_notes:
        parts.append("=== MEMORY (COMPRESSED HISTORY) ===")
        for note in memory_notes[-5:]:
            parts.append(f"• {note}")
        parts.append("")

    # System graph details if present
    if loaded_data and loaded_data.get('system_graph'):
        parts.append("=== SYSTEM ARCHITECTURE (ALWAYS VISIBLE) ===")
        gname = loaded_data['system_graph']['name']
        gdata = loaded_data['system_graph']['data']
        parts.extend(format_graph_for_display(gdata, gname))
        parts.append("")

    if action_log:
        parts.append("=== ACTIONS PERFORMED (SUMMARY) ===")
        for entry in action_log[-10:]:
            act = entry.get('action', '-')
            r = entry.get('result', '')
            rs = (r[:100] if isinstance(r, str) else str(r)[:100])
            parts.append(f"- {act}: {rs}")
        parts.append("")

    return "\n".join(parts)
