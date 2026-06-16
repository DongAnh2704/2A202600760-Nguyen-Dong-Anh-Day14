from typing import Dict, Any
from agents.base_agent import BaseAgent
from src.task9_retrieval_pipeline import retrieve


class LegalResearcherAgent(BaseAgent):
    def __init__(self):
        super().__init__("Legal Researcher")

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        top_k = payload.get("top_k", 5)

        # Truy xuất tài liệu từ pipeline
        sources = retrieve(query, top_k=top_k)
        return {"sources": sources}
