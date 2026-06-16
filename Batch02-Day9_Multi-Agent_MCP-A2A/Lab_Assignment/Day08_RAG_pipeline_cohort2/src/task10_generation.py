"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ bằng chứng để trả lời mà không làm vượt giới hạn token hoặc gây loãng thông tin
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: tạo độ phong phú và tự nhiên vừa phải cho câu trả lời nhưng vẫn giữ được tính chính xác
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.2 vì: Tác vụ RAG pháp luật yêu cầu tính chính xác cao, hạn chế LLM tự sáng tạo (hallucination)
TEMPERATURE = 0.1


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use vietnamese language to answer
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh hiệu ứng "lost in the middle" (quên thông tin ở giữa prompt).

    Chiến thuật: Đưa các chunk có độ liên quan cao nhất lên đầu và xuống cuối prompt,
    các chunk ít liên quan hơn xếp ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]

    Args:
        chunks: Danh sách các chunk đã sắp xếp giảm dần theo điểm số

    Returns:
        Danh sách các chunk sau khi được sắp xếp lại.
    """
    if len(chunks) <= 2:
        return chunks

    # Tách các chỉ mục chẵn và lẻ
    even_indices = [chunks[i] for i in range(len(chunks)) if i % 2 == 0]
    odd_indices = [chunks[i] for i in range(len(chunks)) if i % 2 != 0]

    # Đảo ngược danh sách lẻ (các kết quả tốt nhì, tốt tư...) để xếp ở cuối
    return even_indices + odd_indices[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành chuỗi ngữ cảnh để đưa vào Prompt.
    Mỗi chunk có ghi rõ nguồn (source) để LLM có thể dễ dàng trích dẫn.

    Args:
        chunks: Danh sách các chunk

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Tài liệu {i}")
        doc_type = chunk.get("metadata", {}).get("type", "không rõ")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K, **retrieve_kwargs) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số lượng kết quả cuối cùng
        **retrieve_kwargs: Tham số bổ sung truyền cho hàm retrieve (ví dụ: use_reranking)

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Bước 1: Truy vấn các tài liệu liên quan
    chunks = retrieve(query, top_k=top_k, **retrieve_kwargs)

    # Bước 2: Sắp xếp lại để tránh hiện tượng trôi thông tin
    reordered_chunks = reorder_for_llm(chunks)

    # Bước 3: Định dạng ngữ cảnh
    context = format_context(reordered_chunks)

    # Bước 4: Tạo Prompt cho User
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Bước 5: Gọi LLM (OpenAI) hoặc chạy cơ chế giả lập Mock nếu không có API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    if not api_key or api_key.startswith("sk-xxx"):
        print("⚠ Cảnh báo: OPENAI_API_KEY trống hoặc không hợp lệ. Đang chạy ở chế độ giả lập (Mock LLM).")
        # Sinh câu trả lời giả lập kèm trích dẫn nguồn
        answer = get_mock_llm_response(query, chunks)
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"❌ Lỗi khi gọi OpenAI API: {e}. Tự động fallback sang chế độ giả lập...")
            answer = get_mock_llm_response(query, chunks)

    # Bước 6: Trả về kết quả
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


def get_mock_llm_response(query: str, chunks: list[dict]) -> str:
    """Tự động tạo câu trả lời tổng hợp thô kèm trích dẫn khi thiếu API key."""
    if not chunks:
        return "Tôi không thể tìm thấy tài liệu phù hợp trong cơ sở dữ liệu để trả lời câu hỏi này."

    response = (
        f"Dưới đây là thông tin trích xuất liên quan đến câu hỏi '{query}' "
        "từ cơ sở dữ liệu (Chế độ giả lập RAG do chưa cấu hình OpenAI API Key):\n\n"
    )

    for i, chunk in enumerate(chunks[:3], 1):
        source = chunk.get("metadata", {}).get("source", "Tài liệu gốc")
        doc_name = source.replace(".md", "").replace("_", " ")
        content_lines = chunk["content"].strip().split("\n")
        # Lấy dòng dài nhất có ý nghĩa trong chunk để hiển thị
        best_line = max(content_lines, key=len) if content_lines else chunk["content"]
        if len(best_line) > 200:
            best_line = best_line[:200] + "..."
            
        response += f"- [{doc_name}]: {best_line} [Tham chiếu: {source}, Đoạn số {chunk.get('metadata', {}).get('chunk_index', i)}]\n\n"

    response += "\n*(Vui lòng thiết lập khóa OPENAI_API_KEY trong file `.env` để sinh câu trả lời tự nhiên bằng mô hình GPT)*"
    return response


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
