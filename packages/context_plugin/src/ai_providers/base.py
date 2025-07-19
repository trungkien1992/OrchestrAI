from abc import ABC, abstractmethod
from typing import Any, Dict


class AIProvider(ABC):
    """Abstract base class for AI API providers."""

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Run a completion/generation request."""
        pass

    @abstractmethod
    async def chat(self, messages: list, **kwargs) -> str:
        """Run a chat-based request (if supported)."""
        pass

    @abstractmethod
    def validate_key(self) -> bool:
        """Validate the API key (optional, can be a no-op)."""
        pass
