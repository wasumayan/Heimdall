"""
Anthropic Claude provider implementation.
"""

from __future__ import annotations

import json
import os
import time
from typing import TypeVar

from pydantic import BaseModel

from .base_provider import BaseLLMProvider
from .schema_definitions import get_schema_definition

T = TypeVar('T', bound=BaseModel)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: int = 120,
        retries: int = 3,
        verbose: bool = False,
        thinking_enabled: bool = False,
        **kwargs  # Accept additional kwargs for compatibility
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            model_name: Model identifier (e.g., claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022)
            api_key: API key (optional if using env var)
            api_key_env: Environment variable name for API key
            timeout: Request timeout in seconds
            retries: Number of retry attempts
            verbose: Enable verbose logging
            thinking_enabled: Enable thinking mode (for supported models)
        """
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
        self.thinking_enabled = thinking_enabled
        self._last_token_usage = None
        
        # Get API key
        if api_key:
            self.api_key = api_key
        elif api_key_env:
            self.api_key = os.getenv(api_key_env)
            if not self.api_key:
                raise ValueError(f"API key not found in environment variable {api_key_env}")
        else:
            raise ValueError("Either api_key or api_key_env must be provided")
        
        # Initialize client
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.client = Anthropic(api_key=self.api_key)
    
    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        """Make a structured call using Claude's structured output."""
        # Get schema definition from centralized source
        schema_info = get_schema_definition(schema)
        
        # Add schema info to user prompt
        full_user_prompt = f"{user}\n\n{schema_info}\n\nReturn ONLY valid JSON, no markdown or explanation."
        
        # Log request details
        request_chars = len(system) + len(full_user_prompt)
        if self.verbose:
            print("\n[Anthropic Request]")
            print(f"  Model: {self.model_name}")
            try:
                schema_name = getattr(schema, "__name__", str(schema))
            except Exception:
                schema_name = str(schema)
            print(f"  Schema: {schema_name}")
            print(f"  Total prompt: {request_chars:,} chars (~{request_chars//4:,} tokens)")
        
        last_err = None
        
        for attempt in range(self.retries):
            try:
                start_time = time.time()
                
                # Make API call
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.7,
                    system=system,
                    messages=[
                        {"role": "user", "content": full_user_prompt}
                    ]
                )
                
                # Extract JSON from response
                response_text = response.content[0].text if response.content else ""
                
                # Track token usage
                if hasattr(response, 'usage'):
                    self._last_token_usage = {
                        'input_tokens': response.usage.input_tokens or 0,
                        'output_tokens': response.usage.output_tokens or 0,
                        'total_tokens': (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
                    }
                
                # Log response details
                response_time = time.time() - start_time
                if self.verbose:
                    print("[Anthropic Response]")
                    print(f"  Time: {response_time:.2f}s")
                    print(f"  Output: {len(response_text):,} chars")
                    if hasattr(response, 'usage'):
                        print(f"  Input tokens: {response.usage.input_tokens:,}")
                        print(f"  Output tokens: {response.usage.output_tokens:,}")
                
                # Try to extract JSON from the response
                # Claude sometimes wraps JSON in markdown code blocks
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    json_str = response_text[start:end].strip()
                elif "```" in response_text:
                    start = response_text.find("```") + 3
                    end = response_text.find("```", start)
                    json_str = response_text[start:end].strip()
                else:
                    json_str = response_text.strip()
                
                # Parse JSON
                json_data = json.loads(json_str)
                
                # Convert to Pydantic model
                return schema(**json_data)
                
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retries - 1:
                    wait_time = 2 ** attempt
                    if self.verbose:
                        print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise RuntimeError(f"Failed after {self.retries} attempts: {last_err}")
    
    def raw(self, *, system: str, user: str) -> str:
        """Make a raw text call."""
        # Log request details
        request_chars = len(system) + len(user)
        if self.verbose:
            print("\n[Anthropic Request]")
            print(f"  Model: {self.model_name}")
            print(f"  Total prompt: {request_chars:,} chars (~{request_chars//4:,} tokens)")
        
        last_err = None
        
        for attempt in range(self.retries):
            try:
                start_time = time.time()
                
                # Make API call
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.7,
                    system=system,
                    messages=[
                        {"role": "user", "content": user}
                    ]
                )
                
                # Extract text from response
                response_text = response.content[0].text if response.content else ""
                
                # Track token usage
                if hasattr(response, 'usage'):
                    self._last_token_usage = {
                        'input_tokens': response.usage.input_tokens or 0,
                        'output_tokens': response.usage.output_tokens or 0,
                        'total_tokens': (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
                    }
                
                # Log response details
                response_time = time.time() - start_time
                if self.verbose:
                    print("[Anthropic Response]")
                    print(f"  Time: {response_time:.2f}s")
                    print(f"  Output: {len(response_text):,} chars")
                    if hasattr(response, 'usage'):
                        print(f"  Input tokens: {response.usage.input_tokens:,}")
                        print(f"  Output tokens: {response.usage.output_tokens:,}")
                
                return response_text
                
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retries - 1:
                    wait_time = 2 ** attempt
                    if self.verbose:
                        print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise RuntimeError(f"Failed after {self.retries} attempts: {last_err}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Anthropic"
    
    @property
    def supports_thinking(self) -> bool:
        """Check if this model supports thinking mode."""
        # Claude 3.5 Sonnet supports thinking through o1-style reasoning
        return self.thinking_enabled and "3-5-sonnet" in self.model_name.lower()
    
    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return token usage from the last call if available."""
        return self._last_token_usage
