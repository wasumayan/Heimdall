"""Scout role (junior agent).

Phase 1/2 refactor shim: exposes `Scout` while preserving the
existing AutonomousAgent implementation from agent_core for test stability.
"""

from .agent_core import AutonomousAgent as Scout  # re-export

__all__ = ["Scout"]
