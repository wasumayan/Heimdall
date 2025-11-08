"""LLM client wrapper for structured outputs - supports multiple providers."""
from __future__ import annotations

# Re-export the unified client as LLMClient for backward compatibility
from .unified_client import T, UnifiedLLMClient as LLMClient


# Also export the error class for backward compatibility
class StructuredCallError(RuntimeError):
    """Error during structured LLM call."""
    pass


# The old LLMClient implementation is now replaced by UnifiedLLMClient
# which supports multiple providers (OpenAI, Gemini, etc.)
# This file is kept for backward compatibility
__all__ = ['LLMClient', 'StructuredCallError', 'T']