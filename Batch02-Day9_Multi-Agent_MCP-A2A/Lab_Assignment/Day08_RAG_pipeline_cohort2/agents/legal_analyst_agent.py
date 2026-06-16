import os
import requests
from typing import Dict, Any
from agents.base_agent import BaseAgent


class LegalAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__("Legal Analyst")

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        sources = payload.get("sources", [])
        is_local_mode = payload.get("is_local_mode", False)

        context_str = "\n\n".join([
            f"[Tài liệu {i+1} | Nguồn: {s.get('metadata', {}).get('source', 'chưa rõ')}]\n{s.get('content', '')}"
            for i, s in enumerate(sources)
        ])

        system_prompt = (
            "Bạn là một chuyên gia phân tích luật (Legal Analyst) trong hệ thống Multi-Agent pháp luật ma túy.\n"
            "Nhiệm vụ của bạn là đọc các tài liệu ngữ cảnh pháp lý và câu hỏi, sau đó phân tích và viết một bản thảo câu trả lời (draft answer) chi tiết, dễ hiểu bằng tiếng Việt.\n"
            "BẮT BUỘC: Trả lời hoàn toàn bằng TIẾNG VIỆT. Tuyệt đối KHÔNG sử dụng tiếng Trung Quốc (chữ Hán) hay tiếng Anh trong câu trả lời."
        )

        user_prompt = (
            f"Dựa vào các tài liệu ngữ cảnh dưới đây để soạn thảo câu trả lời nháp cho câu hỏi.\n\n"
            f"Tài liệu ngữ cảnh:\n{context_str}\n\n"
            f"Câu hỏi của người dùng: \"{query}\"\n\n"
            f"Bản thảo câu trả lời nháp:"
        )

        draft_answer = ""

        if is_local_mode:
            try:
                ollama_url = "http://localhost:11434/api/chat"
                req_payload = {
                    "model": "qwen2.5:7b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False
                }
                resp = requests.post(ollama_url, json=req_payload, timeout=90)
                resp.raise_for_status()
                draft_answer = resp.json().get("message", {}).get("content", "")
            except Exception:
                pass
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key and not api_key.startswith("sk-xxx"):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.4,
                        max_tokens=800
                    )
                    draft_answer = response.choices[0].message.content
                except Exception:
                    pass

        # Fallback dùng Mock thô nếu cả hai cách gọi model đều thất bại
        if not draft_answer:
            draft_answer = self._get_mock_draft(query, sources)

        return {"draft_answer": draft_answer}

    def _get_mock_draft(self, query: str, sources: list) -> str:
        if not sources:
            return "Tôi không tìm thấy tài liệu liên quan để phân tích."
        content_snippet = sources[0].get("content", "")[:200]
        source_name = sources[0].get("metadata", {}).get("source", "Tài liệu gốc")
        return (
            f"[Bản thảo phân tích của Legal Analyst cho câu hỏi '{query}']:\n"
            f"Dựa trên tài liệu '{source_name}': {content_snippet}..."
        )
