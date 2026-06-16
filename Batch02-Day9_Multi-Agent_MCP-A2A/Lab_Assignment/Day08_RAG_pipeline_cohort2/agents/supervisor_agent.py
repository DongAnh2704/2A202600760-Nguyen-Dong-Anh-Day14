from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from agents.router_agent import RouterAgent
from agents.general_agent import GeneralAgent
from agents.legal_researcher_agent import LegalResearcherAgent
from agents.legal_analyst_agent import LegalAnalystAgent
from agents.fact_checker_agent import FactCheckerAgent


class SupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Supervisor Agent")
        # Khởi tạo các Worker dưới quyền quản lý của Supervisor
        self.router_worker = RouterAgent()
        self.general_worker = GeneralAgent()
        self.researcher_worker = LegalResearcherAgent()
        self.analyst_worker = LegalAnalystAgent()
        self.fact_checker_worker = FactCheckerAgent()

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt", "")
        is_local_mode = payload.get("is_local_mode", False)
        top_k = payload.get("top_k", 5)

        trace = [self.name]

        # Bước 1: Supervisor gọi Router để phân loại câu hỏi
        trace.append(self.router_worker.name)
        router_res = self.router_worker.handle({"prompt": prompt, "is_local_mode": is_local_mode})
        classification = router_res.get("classification", "LEGAL")

        # Bước 2: Phân phối công việc cho các Worker phù hợp
        if classification == "GENERAL":
            # Câu hỏi xã giao hoặc ngoài lề -> Chuyển General Agent xử lý
            trace.append(self.general_worker.name)
            general_res = self.general_worker.handle({"prompt": prompt, "is_local_mode": is_local_mode})
            return {
                "answer": general_res.get("answer", ""),
                "sources": [],
                "retrieval_source": "none",
                "agent_trace": trace
            }
        else:
            # Câu hỏi pháp luật -> Chạy quy trình RAG Multi-Agent tuần tự
            
            # 2.1: Gọi Researcher để tìm kiếm tài liệu
            trace.append(self.researcher_worker.name)
            researcher_res = self.researcher_worker.handle({"query": prompt, "top_k": top_k})
            sources = researcher_res.get("sources", [])
            
            # Xác định phương thức tìm kiếm từ nguồn đầu tiên tìm thấy
            retrieval_source = "hybrid"
            if sources:
                retrieval_source = sources[0].get("source", "hybrid")

            # 2.2: Gọi Analyst để viết câu trả lời nháp
            trace.append(self.analyst_worker.name)
            analyst_payload = {"query": prompt, "sources": sources, "is_local_mode": is_local_mode}
            analyst_res = self.analyst_worker.handle(analyst_payload)
            draft_answer = analyst_res.get("draft_answer", "")

            # 2.3: Gọi Fact Checker để kiểm chứng và chèn citation
            trace.append(self.fact_checker_worker.name)
            fact_checker_payload = {
                "query": prompt,
                "sources": sources,
                "draft_answer": draft_answer,
                "is_local_mode": is_local_mode
            }
            fact_checker_res = self.fact_checker_worker.handle(fact_checker_payload)
            final_answer = fact_checker_res.get("final_answer", "")

            return {
                "answer": final_answer,
                "sources": sources,
                "retrieval_source": retrieval_source,
                "agent_trace": trace
            }
