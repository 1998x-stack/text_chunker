from __future__ import annotations


class ChunkerError(Exception):
    """Base exception for all text-chunker errors."""
    pass


class ConfigError(ChunkerError):
    """Raised when configuration is invalid or missing."""
    pass


class ProviderError(ChunkerError):
    """Raised when an LLM or embedding provider fails."""
    pass
