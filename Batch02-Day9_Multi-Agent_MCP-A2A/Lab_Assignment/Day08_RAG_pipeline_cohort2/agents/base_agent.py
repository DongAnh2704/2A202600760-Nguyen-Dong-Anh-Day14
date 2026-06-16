from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Simple abstract base class for agents."""

    def __init__(self, name: str):
        self.name = name
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    @abstractmethod
    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming request and return a dict response."""
        raise NotImplementedError()
