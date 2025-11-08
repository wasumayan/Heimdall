"""CLI utility functions."""

import os
import sys


def get_cli_command() -> str:
    """Get the actual CLI command used to run this script.
    
    Returns:
        The command string that should be used in help text,
        e.g., "./hound.py", "python hound.py", or "hound"
    """
    if not sys.argv:
        return "hound"
    
    cli_cmd = sys.argv[0]
    
    # If run as ./script.py, keep as is
    if cli_cmd.startswith('./'):
        return cli_cmd
    
    # If it's a .py file not run with ./, add python prefix
    cli_cmd_base = os.path.basename(cli_cmd)
    if cli_cmd_base.endswith('.py'):
        return f"python {cli_cmd_base}"
    
    # Otherwise use the base name (e.g., installed command)
    return cli_cmd_base or "hound"