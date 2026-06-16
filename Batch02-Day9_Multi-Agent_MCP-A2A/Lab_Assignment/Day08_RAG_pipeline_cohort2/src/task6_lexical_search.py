"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
"""

import os
import pickle
from rank_bm25 import BM25Okapi

# File lưu trữ corpus chunks đã phân mảnh từ Task 4
CHUNKS_PATH = "data/faiss/chunks.pkl"

# Lazy-loaded variables
_corpus = None
_bm25 = None


def get_corpus() -> list[dict]:
    """Tải corpus đã chunk ở Task 4."""
    global _corpus
    if _corpus is not None:
        return _corpus

    if os.path.exists(CHUNKS_PATH):
        try:
            with open(CHUNKS_PATH, "rb") as f:
                _corpus = pickle.load(f)
            # Thêm thông báo gọn gàng
        except Exception as e:
            print(f"❌ Lỗi khi đọc file corpus: {e}")
            _corpus = []
    else:
        _corpus = []
    return _corpus


def get_bm25_index():
    """Tạo hoặc lấy BM25 index của corpus."""
    global _bm25
    if _bm25 is not None:
        return _bm25

    corpus = get_corpus()
    if not corpus:
        return None

    # Phân mảnh từ tiếng Việt đơn giản (tách từ viết thường)
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25 = get_bm25_index()
    corpus = get_corpus()

    if not bm25 or not corpus:
        print("⚠️ Cảnh báo: BM25 index chưa được xây dựng (chưa chạy Task 4 hoặc không có dữ liệu).")
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy ra top_k index có điểm số cao nhất
    import numpy as np
    # Sắp xếp các chỉ mục từ lớn đến bé
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # Chỉ lấy các chunk có độ trùng lặp từ khóa (score > 0)
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"]
            })
            
    return results


if __name__ == "__main__":
    # Test
    print("Đang chạy thử tìm kiếm từ khóa BM25...")
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    if results:
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    else:
        print("Không tìm thấy kết quả phù hợp (Hãy đảm bảo Task 4 đã chạy thành công).")
