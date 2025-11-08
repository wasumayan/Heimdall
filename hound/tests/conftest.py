"""Pytest configuration to resolve local LLm modules safely.

Some environments have a global `llm` package installed that can shadow our
local `hound/llm` package. To ensure tests import the in-repo implementation,
we explicitly alias `llm` and its submodules to `hound.llm.*` in sys.modules
before test collection.
"""

import importlib
import sys
from pathlib import Path

# Ensure current directory (package root) is at the front of sys.path
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

try:
    # Force-map the top-level name 'llm' to our local implementation 'hound.llm'
    sys.modules['llm'] = importlib.import_module('hound.llm')
    # Preload and alias common submodules to ensure consistent resolution
    for _sub in [
        'client', 'unified_client', 'openai_provider', 'token_tracker',
        'gemini_provider', 'anthropic_provider', 'deepseek_provider', 'xai_provider',
        'schemas', 'schema_definitions', 'tokenization', 'mock_provider'
    ]:
        try:
            sys.modules[f'llm.{_sub}'] = importlib.import_module(f'hound.llm.{_sub}')
        except Exception:
            pass
except Exception:
    # Fall back silently; tests will raise clearer errors later
    pass
