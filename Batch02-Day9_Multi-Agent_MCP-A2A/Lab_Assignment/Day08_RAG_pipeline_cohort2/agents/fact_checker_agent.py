import os
import requests
from typing import Dict, Any
from agents.base_agent import BaseAgent


class FactCheckerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Fact Checker")

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        sources = payload.get("sources", [])
        draft_answer = payload.get("draft_answer", "")
        is_local_mode = payload.get("is_local_mode", False)

        context_str = "\n\n".join([
            f"[Tài liệu {i+1} | Nguồn: {s.get('metadata', {}).get('source', 'chưa rõ')}]\n{s.get('content', '')}"
            for i, s in enumerate(sources)
        ])

        system_prompt = (
            "Bạn là một chuyên gia kiểm chứng thông tin pháp luật (Fact Checker/Critic) trong hệ thống Multi-Agent.\n"
            "Nhiệm vụ của bạn là đọc các tài liệu ngữ cảnh gốc, so sánh với bản thảo câu trả lời (draft_answer), sau đó hiệu chỉnh để tạo ra câu trả lời cuối cùng chính xác và chuyên nghiệp.\n\n"
            "Quy tắc kiểm chứng:\n"
            "1. Tuyệt đối loại bỏ thông tin không được đề cập trong tài liệu gốc (hallucination).\n"
            "2. Với mỗi luận điểm hoặc thông tin thực tế, bạn BẮT BUỘC phải đặt trích dẫn dạng móc vuông tham chiếu đến nguồn tài liệu gốc (Ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [Nguồn A, Điều B]).\n"
            "3. Nếu thông tin không đủ để kết luận, nói rõ 'Tôi không thể xác minh thông tin này từ nguồn hiện có'.\n"
            "4. BẮT BUỘC: Trả lời hoàn toàn bằng TIẾNG VIỆT. Tuyệt đối KHÔNG sử dụng tiếng Trung Quốc (chữ Hán) hay tiếng Anh trong câu trả lời."
        )

        user_prompt = (
            f"Hãy đối chiếu tài liệu ngữ cảnh dưới đây với bản thảo nháp để tạo câu trả lời hoàn thiện có trích dẫn chính xác.\n\n"
            f"Tài liệu ngữ cảnh gốc:\n{context_str}\n\n"
            f"Bản thảo câu trả lời nháp:\n{draft_answer}\n\n"
            f"Câu hỏi của người dùng: \"{query}\"\n\n"
            f"Câu trả lời cuối cùng hoàn thiện (có citation):"
        )

        final_answer = ""

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
                final_answer = resp.json().get("message", {}).get("content", "")
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
                        temperature=0.1,
                        max_tokens=800
                    )
                    final_answer = response.choices[0].message.content
                except Exception:
                    pass

        # Fallback tạo trích dẫn thô nếu model lỗi
        if not final_answer:
            final_answer = self._get_mock_final(draft_answer, sources)

        return {"final_answer": final_answer}

    def _get_mock_final(self, draft_answer: str, sources: list) -> str:
        if not sources:
            return draft_answer
        source_name = sources[0].get("metadata", {}).get("source", "Tài liệu gốc")
        return f"{draft_answer}\n\n*(Nguồn tham chiếu: [{source_name}])*"
