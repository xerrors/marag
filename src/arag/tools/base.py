"""Base tool class for ARAG."""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Return OpenAI Function Schema."""
        pass

    @abstractmethod
    def execute(self, context: "AgentContext", **kwargs) -> tuple[str, dict[str, Any]]:
        """Execute the tool and return (result, log)."""
        pass
