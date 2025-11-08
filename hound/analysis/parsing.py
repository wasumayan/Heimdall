"""Robust parsing helpers for agent JSON decisions."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_agent_decision_fallback(response: str) -> dict[str, Any] | None:
    """Attempt to salvage a JSON object from a free-form LLM response.

    Returns a dict or None.
    """
    try:
        # Try direct JSON
        return json.loads(response)
    except Exception:
        pass

    try:
        # Extract first JSON object
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if not m:
            return None
        json_str = m.group()
        # Fix common trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json.loads(json_str)
    except Exception:
        return None

