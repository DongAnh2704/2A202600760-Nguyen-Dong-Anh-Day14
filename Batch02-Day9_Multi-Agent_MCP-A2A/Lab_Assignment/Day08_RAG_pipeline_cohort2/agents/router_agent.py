import os
import requests
from typing import Dict, Any
from agents.base_agent import BaseAgent


class RouterAgent(BaseAgent):
    def __init__(self):
        super().__init__("Router Agent")

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt", "").strip()
        is_local_mode = payload.get("is_local_mode", False)

        system_prompt = (
            "Bạn là Router Agent trong hệ thống Multi-Agent pháp luật ma túy.\n"
            "Nhiệm vụ của bạn là phân loại yêu cầu của người dùng thành một trong hai loại:\n"
            "- LEGAL: Nếu câu hỏi liên quan đến pháp luật ma túy, hình phạt ma túy, tội phạm ma túy hoặc tin tức xã hội về ma túy.\n"
            "- GENERAL: Nếu câu hỏi là chào hỏi, chitchat xã giao, hoặc các câu hỏi không liên quan đến pháp luật ma túy (ví dụ: thời tiết, lập trình, nấu ăn, toán học...).\n\n"
            "Chỉ trả về duy nhất một từ: 'LEGAL' hoặc 'GENERAL'. Không trả thêm bất kỳ giải thích nào."
        )

        classification = "LEGAL"  # Mặc định

        if is_local_mode:
            try:
                ollama_url = "http://localhost:11434/api/chat"
                req_payload = {
                    "model": "qwen2.5:7b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Câu hỏi: \"{prompt}\"\nPhân loại:"}
                    ],
                    "stream": False
                }
                resp = requests.post(ollama_url, json=req_payload, timeout=20)
                resp.raise_for_status()
                response_text = resp.json().get("message", {}).get("content", "").strip().upper()
                if "GENERAL" in response_text:
                    classification = "GENERAL"
            except Exception:
                # Fallback dùng rule-based cơ bản nếu Ollama lỗi
                classification = self._rule_based_classify(prompt)
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
                            {"role": "user", "content": f"Câu hỏi: \"{prompt}\""}
                        ],
                        temperature=0.0,
                        max_tokens=5
                    )
                    response_text = response.choices[0].message.content.strip().upper()
                    if "GENERAL" in response_text:
                        classification = "GENERAL"
                except Exception:
                    classification = self._rule_based_classify(prompt)
            else:
                classification = self._rule_based_classify(prompt)

        return {"classification": classification}

    def _rule_based_classify(self, prompt: str) -> str:
        # Nhận diện một số câu chào hỏi xã giao hoặc lạc đề cơ bản
        keywords = [
            "chào", "hello", "hi", "tên gì", "là ai", "chơi", "weather", 
            "thời tiết", "code", "lập trình", "nấu ăn", "ăn gì", "món ăn"
        ]
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in keywords):
            return "GENERAL"
        return "LEGAL"
