"""Compatibility wrapper for Agent C.

All functionality now lives in `agents.network.auth_endpoint_agent`. This stub
is kept so older imports keep working without change.
"""
from .auth_endpoint_agent import (  # noqa: F401
    AgentCConfig,
    AuthEndpointAgent,
    AuthEndpointScanResult,
    BrowserCaptureConfig,
    PlaywrightAuthEndpointAgent,
    run_cli,
)

__all__ = [
    "AgentCConfig",
    "AuthEndpointAgent",
    "AuthEndpointScanResult",
    "BrowserCaptureConfig",
    "PlaywrightAuthEndpointAgent",
    "run_cli",
]
