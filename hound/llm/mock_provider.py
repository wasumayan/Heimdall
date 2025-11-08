"""Mock LLM provider for testing."""
from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from .base_provider import BaseLLMProvider

T = TypeVar('T', bound=BaseModel)


class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing purposes."""
    
    def __init__(self, config: dict[str, Any], model_name: str, **kwargs):
        """Initialize mock provider."""
        self.config = config
        self.model_name = model_name
        self.call_count = 0
        self.responses = []
        self.response_index = 0
        self._last_token_usage = None
        
        # Store mock instance if provided for testing
        self._mock_instance = kwargs.get('mock_instance')
        
    def set_responses(self, responses):
        """Set predefined responses for testing."""
        self.responses = responses
        self.response_index = 0
        
    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        """Return structured response based on mock configuration."""
        self.call_count += 1
        
        # Set mock token usage
        self._last_token_usage = {
            'input_tokens': 100,
            'output_tokens': 50,
            'total_tokens': 150
        }
        
        # If we have a mock instance with parse method, use it
        if self._mock_instance and hasattr(self._mock_instance, 'parse'):
            return self._mock_instance.parse(system=system, user=user, schema=schema)
        
        # If predefined responses are set, use them
        if self.responses and self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            
            if isinstance(response, dict):
                return schema(**response)
            elif isinstance(response, BaseModel):
                return response
            else:
                # Try to parse as JSON
                try:
                    data = json.loads(response) if isinstance(response, str) else response
                    return schema(**data)
                except Exception:
                    # Return a minimal valid instance
                    return self._create_minimal_instance(schema)
        
        # Default: create minimal valid instance
        return self._create_minimal_instance(schema)
    
    def raw(self, *, system: str, user: str) -> str:
        """Return raw text response."""
        self.call_count += 1
        
        # Set mock token usage
        self._last_token_usage = {
            'input_tokens': 100,
            'output_tokens': 50,
            'total_tokens': 150
        }
        
        # If we have a mock instance with raw method, use it
        if self._mock_instance and hasattr(self._mock_instance, 'raw'):
            return self._mock_instance.raw(system=system, user=user)
        
        # If predefined responses are set, use them
        if self.responses and self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            
            if isinstance(response, str):
                return response
            elif isinstance(response, dict):
                return json.dumps(response)
            else:
                return str(response)
        
        # Default response based on context
        if 'report' in system.lower() or 'report' in user.lower():
            return "# Security Analysis Report\n\n## Summary\nMock report generated for testing."
        elif 'finalize' in system.lower() or 'verdict' in user.lower():
            return json.dumps({
                "verdict": "confirmed",
                "reasoning": "Mock confirmation for testing",
                "confidence": 0.95
            })
        else:
            return "Mock response for testing"
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "mock"
    
    @property
    def supports_thinking(self) -> bool:
        """Mock provider doesn't support thinking mode."""
        return False
    
    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return mock token usage."""
        return self._last_token_usage
    
    def _create_minimal_instance(self, schema: type[T]) -> T:
        """Create a minimal valid instance of the schema."""
        # This is a simplified version - in real tests you'd want more control
        fields = {}
        
        # Handle special cases based on schema name
        if hasattr(schema, '__name__'):
            if schema.__name__ == 'AgentDecision':
                fields = {
                    'action': 'complete',
                    'reasoning': 'Default mock action',
                    'parameters': {}
                }
            elif schema.__name__ == 'FinalizationResult':
                fields = {
                    'verdict': 'confirmed',
                    'reasoning': 'Mock confirmation',
                    'confidence': 0.95
                }
        
        # Try to create instance with minimal fields
        try:
            return schema(**fields)
        except Exception:
            # If that fails, try with empty dict
            try:
                return schema()
            except Exception:
                # Last resort - raise error
                raise ValueError(f"Cannot create mock instance of {schema}")
