import os as _os
from functools import lru_cache

from rich.console import Console

console = Console()
_VERBOSE = _os.environ.get("HOUND_LLM_VERBOSE", "").lower() in {"1","true","yes","on"}

class TokenCounter:
    """
    Token counter with perfect accuracy for OpenAI/Anthropic, fallback for others.
    """

    def __init__(self):
        # Tokenizer libraries - loaded on demand
        self._tiktoken = None
        self._anthropic = None

        # Cache for tiktoken encoders
        self._encoder_cache: dict[str, any] = {}

        # Try to load tokenizer libraries
        self._tiktoken_available = self._try_import_tiktoken()
        self._anthropic_available = self._try_import_anthropic()

        if _VERBOSE:
            console.print(
                f"TokenCounter: tiktoken={self._tiktoken_available}, anthropic={self._anthropic_available}"
            )

    def _try_import_tiktoken(self) -> bool:
        """Import tiktoken for OpenAI tokenization."""
        try:
            import tiktoken

            self._tiktoken = tiktoken
            return True
        except ImportError:
            if _VERBOSE:
                console.print("tiktoken not available - OpenAI token counting will use approximation")
            return False

    def _try_import_anthropic(self) -> bool:
        """Import anthropic for Claude tokenization."""
        try:
            import anthropic

            self._anthropic = anthropic
            return True
        except ImportError:
            if _VERBOSE:
                console.print(
                    "anthropic library not available - Claude token counting will use approximation"
                )
            return False

    @lru_cache(maxsize=16)
    def _get_openai_encoder(self, model: str):
        """Get tiktoken encoder for OpenAI model."""
        if not self._tiktoken_available:
            return None

        try:
            model_lower = model.lower()

            # GPT-4 family
            if "gpt-4o" in model_lower:
                return self._tiktoken.encoding_for_model("gpt-4o")
            elif "gpt-4" in model_lower:
                return self._tiktoken.encoding_for_model("gpt-4")

            # GPT-3.5 family
            elif "gpt-3.5" in model_lower:
                return self._tiktoken.encoding_for_model("gpt-3.5-turbo")

            # o1 models - use gpt-4 encoding
            elif "o1" in model_lower:
                return self._tiktoken.encoding_for_model("gpt-4")

            # Try exact model name first
            else:
                try:
                    return self._tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fallback to cl100k_base (most common)
                    return self._tiktoken.get_encoding("cl100k_base")

        except Exception as e:
            console.print(f"Failed to get OpenAI encoder for {model}: {e}")
            return None

    def _count_openai_tokens(self, text: str, model: str) -> int:
        """Count tokens for OpenAI models using tiktoken."""
        encoder = self._get_openai_encoder(model)
        if encoder is None:
            # Fallback to approximation
            return max(1, len(text) // 4)

        try:
            return len(encoder.encode(text))
        except Exception as e:
            console.print(f"OpenAI tokenization failed for {model}: {e}")
            return max(1, len(text) // 4)

    def _count_anthropic_tokens(self, text: str, model: str) -> int:
        """Count tokens for Anthropic models."""
        if not self._anthropic_available:
            # Fallback to approximation
            return max(1, len(text) // 4)

        try:
            # Use Anthropic's token counting
            client = self._anthropic.Anthropic()
            count_response = client.messages.count_tokens(
                model=model, messages=[{"role": "user", "content": text}]
            )
            return count_response.input_tokens

        except Exception as e:
            console.print(f"Anthropic tokenization failed for {model}: {e}")
            # Fallback to approximation
            return max(1, len(text) // 4)

    def count_tokens(self, text: str, provider: str, model: str) -> int:
        """
        Count tokens using provider-specific method or fallback.

        Args:
            text: Text to tokenize
            provider: LLM provider (openai, anthropic, gemini, xai)
            model: Specific model name

        Returns:
            Number of tokens
        """
        if not text or not text.strip():
            return 0

        provider_lower = provider.lower()

        try:
            if provider_lower == "openai":
                return self._count_openai_tokens(text, model)
            elif provider_lower == "anthropic":
                return self._count_anthropic_tokens(text, model)
            else:
                # Fallback for gemini, xai, and unknown providers
                return max(1, len(text) // 4)

        except Exception as e:
            console.print(f"Token counting failed for {provider}:{model}: {e}")
            return max(1, len(text) // 4)

    def get_diagnostics(self) -> dict[str, any]:
        """Get tokenization system diagnostics."""
        return {
            "tiktoken_available": self._tiktoken_available,
            "anthropic_available": self._anthropic_available,
            "encoder_cache_size": len(self._encoder_cache),
            "accurate_providers": (
                ["openai", "anthropic"]
                if self._tiktoken_available and self._anthropic_available
                else (["openai"] if self._tiktoken_available else [])
                + (["anthropic"] if self._anthropic_available else [])
            ),
            "fallback_providers": ["gemini", "xai"],
        }


# Global instance
_token_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """Get global token counter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def count_tokens(text: str, provider: str, model: str) -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count_tokens(text, provider, model)


def get_diagnostics() -> dict[str, any]:
    """Get tokenization diagnostics."""
    return get_token_counter().get_diagnostics()
