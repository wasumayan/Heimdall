"""Base provider interface for LLM clients."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def __init__(self, config: dict[str, Any], model_name: str, **kwargs):
        """Initialize the provider with configuration."""
        pass
    
    @abstractmethod
    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        """
        Make a structured call returning an instance of the schema.
        
        Args:
            system: System prompt
            user: User prompt
            schema: Pydantic model class for structured output
            
        Returns:
            Instance of the schema with parsed data
        """
        pass
    
    @abstractmethod
    def raw(self, *, system: str, user: str) -> str:
        """
        Make a plain text call without structured output.
        
        Args:
            system: System prompt
            user: User prompt
            
        Returns:
            Raw text response
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for logging."""
        pass
    
    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        """Return whether this provider/model supports thinking mode."""
        pass
    
    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return token usage from the last call if available.
        
        Returns:
            Dict with 'input_tokens', 'output_tokens', 'total_tokens' or None
        """
        return None