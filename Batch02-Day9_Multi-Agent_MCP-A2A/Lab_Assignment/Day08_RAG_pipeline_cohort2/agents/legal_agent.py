import os
import requests
import logging
from typing import Dict, Any
from agents.base_agent import BaseAgent
from src.task10_generation import generate_with_citation

logger = logging.getLogger(__name__)


class LegalRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__("Legal RAG Agent")

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt", "")
        is_local_mode = payload.get("is_local_mode", False)
        top_k = payload.get("top_k", 5)

        # Gọi RAG pipeline
        result = generate_with_citation(prompt, top_k=top_k)
        sources = result.get("sources", [])
        retrieval_source = result.get("retrieval_source", "hybrid")

        if is_local_mode:
            context_str = "\n".join([f"- Tài liệu {i+1}: {s.get('content','')}" for i, s in enumerate(sources)])
            system_prompt = (
                "Bạn là một trợ lý pháp luật chuyên nghiệp, chỉ trả lời về Pháp luật phòng chống Ma túy.\n"
                "BẮT BUỘC: Trả lời hoàn toàn bằng TIẾNG VIỆT. Tuyệt đối KHÔNG được sử dụng chữ Hán (tiếng Trung Quốc), chữ Hán Nôm hoặc tiếng Anh."
            )
            user_prompt = (
                f"Dựa vào ngữ cảnh tài liệu dưới đây để trả lời câu hỏi của người dùng:\n"
                f"Ngữ cảnh tài liệu:\n{context_str}\n\n"
                f"Yêu cầu xử lý từ hội thoại:\n{prompt}\n\n"
                f"Câu trả lời của bạn:"
            )
            try:
                ollama_url = "http://localhost:11434/api/chat"
                payload_ollama = {
                    "model": "qwen2.5:7b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False
                }
                resp = requests.post(ollama_url, json=payload_ollama, timeout=90)
                resp.raise_for_status()
                answer = resp.json().get("message", {}).get("content", None)
                if not answer:
                    answer = result.get("answer", "Không nhận được phản hồi từ mô hình local.")
            except Exception as ex:
                logger.debug("Ollama local call failed, falling back to pipeline: %s", ex)
                answer = result.get("answer", "Không nhận được phản hồi.")
        else:
            answer = result.get("answer", "Không nhận được phản hồi.")

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_source": retrieval_source
        }
