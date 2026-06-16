import os
import requests
from typing import Dict, Any
from agents.base_agent import BaseAgent


class GeneralAgent(BaseAgent):
    def __init__(self):
        super().__init__("General Agent")

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt", "")
        is_local_mode = payload.get("is_local_mode", False)

        system_prompt = (
            "Bạn là General Agent trong hệ thống Multi-Agent pháp luật ma túy.\n"
            "Nhiệm vụ của bạn là phản hồi các câu hỏi chào hỏi, chitchat xã giao hoặc lịch sự từ chối "
            "nếu câu hỏi lạc đề (không liên quan đến pháp luật ma túy).\n\n"
            "Quy tắc phản hồi:\n"
            "1. Nếu người dùng chào hỏi, hãy chào hỏi lại lịch sự và tự giới thiệu mình là trợ lý chuyên về pháp luật phòng chống ma túy.\n"
            "2. Nếu người dùng hỏi các câu hỏi lạc đề (nấu ăn, lập trình, thời tiết...), hãy lịch sự từ chối trả lời và hướng người dùng quay lại chủ đề luật ma túy."
        )

        answer = "Xin chào! Tôi là trợ lý pháp luật chuyên giải đáp về chủ đề Luật phòng chống ma túy. Tôi có thể giúp gì cho bạn?"

        if is_local_mode:
            try:
                ollama_url = "http://localhost:11434/api/chat"
                local_system_prompt = system_prompt + "\nBẮT BUỘC: Trả lời hoàn toàn bằng TIẾNG VIỆT. Tuyệt đối KHÔNG sử dụng chữ Hán (tiếng Trung) hoặc tiếng Anh."
                req_payload = {
                    "model": "qwen2.5:7b",
                    "messages": [
                        {"role": "system", "content": local_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
                resp = requests.post(ollama_url, json=req_payload, timeout=90)
                resp.raise_for_status()
                answer = resp.json().get("message", {}).get("content", answer)
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
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    answer = response.choices[0].message.content
                except Exception:
                    pass

        return {
            "answer": answer,
            "sources": [],
            "retrieval_source": "none"
        }
