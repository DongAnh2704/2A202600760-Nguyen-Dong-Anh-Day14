import logging
from typing import Dict, Any
from agents.supervisor_agent import SupervisorAgent

logger = logging.getLogger(__name__)


class AgentManager:
    """Agent manager that delegates query execution to the Supervisor Agent.

    Maintains public interface compatibility while utilizing the Supervisor-Worker pattern under the hood.
    """

    def __init__(self):
        self.supervisor = SupervisorAgent()

    def handle_query(self, prompt: str, is_local_mode: bool, top_k: int = 5) -> Dict[str, Any]:
        """Delegate handling to the Supervisor Agent."""
        try:
            return self.supervisor.handle({
                "prompt": prompt,
                "is_local_mode": is_local_mode,
                "top_k": top_k
            })
        except Exception as exc:
            logger.exception("Error in AgentManager.handle_query")
            return {
                "answer": f"Lỗi hệ thống Multi-Agent: {exc}",
                "sources": [],
                "retrieval_source": "error",
                "agent_trace": ["Supervisor Agent"]
            }
