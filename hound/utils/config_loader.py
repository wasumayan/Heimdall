"""
Centralized configuration loading utility.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Priority order:
    1. Explicitly provided config_path
    2. HOUND_CONFIG environment variable
    3. config.yaml in current directory
    4. config.yaml in hound directory
    5. config.example.yaml in hound directory
    6. Empty dict as fallback
    """
    
    # If explicit path provided, use it
    if config_path and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    
    # Check environment variable
    if os.environ.get('HOUND_CONFIG'):
        env_config = Path(os.environ['HOUND_CONFIG'])
        if env_config.exists():
            with open(env_config) as f:
                return yaml.safe_load(f) or {}
    
    # Try current directory
    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        with open(cwd_config) as f:
            return yaml.safe_load(f) or {}
    
    # Try hound directory (where this module lives)
    hound_dir = Path(__file__).parent.parent
    
    # Try config.yaml in hound directory
    hound_config = hound_dir / "config.yaml"
    if hound_config.exists():
        with open(hound_config) as f:
            return yaml.safe_load(f) or {}
    
    # Fallback to example config
    example_config = hound_dir / "config.example.yaml"
    if example_config.exists():
        with open(example_config) as f:
            return yaml.safe_load(f) or {}
    
    # Last resort: return empty config
    # This allows commands to run with defaults
    return {}