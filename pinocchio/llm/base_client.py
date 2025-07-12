"""Base LLM client interface."""

import abc
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseLLMClient(abc.ABC):
    """Base class for LLM clients."""

    def __init__(self):
        """Initialize base LLM client."""
        self.call_count = 0

    @abc.abstractmethod
    async def complete(self, prompt: str, agent_type: Optional[str] = None) -> str:
        """
        Complete prompt with LLM response.

        Args:
            prompt: Input prompt
            agent_type: Optional agent type for context

        Returns:
            LLM response as string
        """
        pass

    @abc.abstractmethod
    async def complete_structured(
        self, prompt: str, agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete prompt and return structured response.

        Args:
            prompt: Input prompt
            agent_type: Optional agent type for context

        Returns:
            Structured response as dictionary
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "call_count": self.call_count,
            "client_type": self.__class__.__name__,
        }
