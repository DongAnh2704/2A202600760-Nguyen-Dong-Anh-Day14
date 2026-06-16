import os
import sys
from pathlib import Path

# === THÊM ĐOẠN NÀY VÀO ĐỂ ẨN LOG WARNING PHIỀN PHỨC ===
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# =====================================================

import streamlit as st

st.set_page_config(page_title="RAG Chatbot Luật Ma túy", page_icon="⚖️", layout="wide")

# Ẩn các biểu tượng hoạt ảnh/người que (running man) bên cạnh nút Stop
st.markdown(
        """
        <style>
        [data-testid="stStatusWidget"] svg,
        [data-testid="stStatusWidget"] [data-testid="stIcon"],
        [data-testid="stStatusWidget"] div[class*="Spinner"],
        [data-testid="stStatusWidget"] [data-testid="stSpinner"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
)

from dotenv import load_dotenv
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Agent manager (multi-agent integration point)
from agents import manager as AGENT_MANAGER

# Import trực tiếp hàm RAG generation từ Task 10
from src.task10_generation import generate_with_citation
from src.task4_chunking_indexing import CHUNKING_METHOD, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv(ROOT_DIR / ".env")


def build_follow_up_prompt(query: str, history: list[dict]) -> str:
    if not history:
        return query

    history_lines = []
    for i, turn in enumerate(history[-5:], start=1):
        history_lines.append(
            f"Turn {i} - User: {turn['question']}\nAssistant: {turn['answer']}"
        )

    history_text = "\n\n".join(history_lines)
    return (
        f"Conversation history:\n{history_text}\n\n"
        f"Current question: {query}"
    )


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def render_source_card(source: dict, index: int) -> None:
    """Hiển thị nguồn tài liệu dưới dạng Card giao diện đẹp mắt và an toàn với mọi Theme"""
    metadata = source.get("metadata", {})
    source_name = metadata.get("source", "Tài liệu chưa rõ nguồn")
    source_type = metadata.get("type", "Chưa phân loại")
    score = source.get("score", 0.0)
    excerpt = source.get("content", "").strip()

    # Dùng container viền ngoài của Streamlit, tự động thay đổi màu nền và màu chữ theo Theme (Sáng/Tối)
    with st.container(border=True):
        st.markdown(f"##### 📌 Nguồn {index}: {source_name}")
        st.caption(f"Loại: {source_type}  •  Độ liên quan (Score): :red[{score:.3f}]")
        st.markdown(f"*{excerpt}*")


def main() -> None:
    init_session_state()

    # Kiểm tra trạng thái cấu hình OpenAI API Key để hiển thị lên Sidebar
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    is_local_mode = not openai_key

    # --- CỘT TRÁI: THÔNG TIN TRỢ LÝ ---
    sidebar, chat_zone = st.columns([1, 2.5], gap="large")

    with sidebar:
        st.logo("https://cdn-icons-png.flaticon.com/512/3429/3429149.png", icon_image="https://lapphap.vn/Upload/PH%C3%81P-L%C3%9D.jpg")
        st.title("⚖️ Trợ Lý Pháp Luật")
        
        if is_local_mode:
            st.caption("Chế độ: **Mô hình Local Qwen2.5:7b (Ollama)**")
            st.warning("Không tìm thấy `OPENAI_API_KEY`. Đã tự động chuyển sang chạy mô hình Local để tiết kiệm chi phí và bảo mật.")
        else:
            st.caption("🌐 Chế độ: **OpenAI GPT Cloud**")
            st.success("✨ Đã nhận diện `OPENAI_API_KEY`. Hệ thống đang sử dụng LLM Cloud.")

        st.info(
            "💡 **Phạm vi hỗ trợ:**\n"
            "Chuyên giải đáp về Pháp luật phòng chống Ma túy, hình phạt tội phạm ma túy và các tin tức xã hội liên quan."
        )
        
        with st.expander("🛠️ Chi tiết hệ thống", expanded=False):
            st.markdown(
                f"- **Giao diện:** Streamlit Chat UI  \n"
                f"- **Mô hình hiện tại:** {'Qwen2.5:7b (Ollama)' if is_local_mode else 'OpenAI GPT'}  \n"
                f"- **Phân mảnh (Strategy):** `{CHUNKING_METHOD}`  \n"
                f"- **Kích thước (Chunk Size):** `{CHUNK_SIZE}` ký tự  \n"
                f"- **Độ chồng lấp (Overlap):** `{CHUNK_OVERLAP}` ký tự  \n"
                f"- **Kiểm soát phạm vi:** Tự động từ chối câu hỏi không liên quan."
            )
            
        # Nút xóa lịch sử chat nằm gọn gàng bên cột trái
        if st.button("🔄 Xóa lịch sử hội thoại", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.rerun()

    # --- CỘT PHẢI: KHUNG CHÁT CHÍNH ---
    with chat_zone:
        st.subheader("💬 Hộp thoại Tư vấn Pháp luật")
        
        # 1. Hiển thị Lịch sử chat theo dạng dòng thời gian hiện đại (st.chat_message)
        for turn in st.session_state.history:
            with st.chat_message("user", avatar="💬"):
                st.markdown(turn["question"])
                
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(turn["answer"])
                
                # Hiển thị tiến trình Multi-Agent đã chạy
                if turn.get("agent_trace"):
                    trace_str = " ➔ ".join([f"`{name}`" for name in turn["agent_trace"]])
                    st.caption(f"🤖 **Tiến trình Multi-Agent:** {trace_str}")
                    
                if turn.get("retrieval_source") and turn["retrieval_source"] != "error" and turn.get("sources"):
                    with st.expander(f"📚 Xem đầy đủ {len(turn.get('sources', []))} nguồn tài liệu trích dẫn", expanded=False):
                        st.caption(f"Phương thức tìm kiếm: {turn['retrieval_source']}")
                        for i, source in enumerate(turn.get("sources", []), start=1):
                            render_source_card(source, i)

        # 2. Ô nhập nội dung chat cố định ở đáy màn hình
        query = st.chat_input("Nhập câu hỏi của bạn về luật phòng chống ma túy tại đây...")

        if query:
            with st.chat_message("user", avatar="💬"):
                st.markdown(query)

            with st.chat_message("assistant", avatar="⚖️"):
                spinner_msg = "Đang xử lý bằng Qwen2.5:7b Local..." if is_local_mode else "Đang xử lý bằng OpenAI..."
                with st.spinner(spinner_msg):
                    prompt = build_follow_up_prompt(query.strip(), st.session_state.history)
                    agent_trace = []
                    try:
                        result = AGENT_MANAGER.handle_query(prompt, is_local_mode, top_k=5)
                        answer = result.get("answer", "Không nhận được phản hồi.")
                        sources = result.get("sources", [])
                        retrieval_source = result.get("retrieval_source", "hybrid")
                        agent_trace = result.get("agent_trace", [])
                    except Exception as exc:
                        answer = (
                            "⚠ Không thể hoàn thành xử lý câu trả lời.\n\n"
                            f"*Chi tiết lỗi:* {exc}"
                        )
                        sources = []
                        retrieval_source = "error"

                # Hiển thị câu trả lời mới sinh lên màn hình
                st.markdown(answer)
                
                # Hiển thị tiến trình Multi-Agent
                if agent_trace:
                    trace_str = " ➔ ".join([f"`{name}`" for name in agent_trace])
                    st.caption(f"🤖 **Tiến trình Multi-Agent:** {trace_str}")
                    
                if retrieval_source != "error" and sources:
                    with st.expander(f"📚 Xem đầy đủ {len(sources)} nguồn tài liệu trích dẫn", expanded=False):
                        st.caption(f"Phương thức tìm kiếm: {retrieval_source}")
                        for i, source in enumerate(sources, start=1):
                            render_source_card(source, i)

            # Lưu lượt hội thoại mới vào bộ nhớ session_state để duy trì bộ nhớ ngữ cảnh (context)
            st.session_state.history.append(
                {
                    "question": query.strip(),
                    "answer": answer,
                    "sources": sources,
                    "retrieval_source": retrieval_source,
                    "agent_trace": agent_trace,
                }
            )
            
            # Khởi động lại luồng để Streamlit cập nhật đồng bộ giao diện
            st.rerun()


if __name__ == "__main__":
    main()