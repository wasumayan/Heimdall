"""
Robust JSON extraction utilities for LLM responses.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> Any | None:
    """Extract a JSON object from arbitrary text.

    Handles common LLM output patterns:
    - Fenced code blocks ```json { ... } ```
    - Leading/trailing prose
    - Trailing commas before closing braces/brackets
    - Falls back to direct json.loads
    """
    if not isinstance(text, str):
        return None

    # Prefer fenced JSON blocks
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m:
        s = m.group(1)
        try:
            return json.loads(s)
        except Exception:
            pass

    # Try to find first balanced JSON object
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    cand = text[start:i+1]
                    # Clean trailing commas before closing
                    cand = re.sub(r',\s*([}\]])', r'\1', cand)
                    try:
                        return json.loads(cand)
                    except Exception:
                        break

    # Direct parse as last resort
    try:
        return json.loads(text)
    except Exception:
        return None

