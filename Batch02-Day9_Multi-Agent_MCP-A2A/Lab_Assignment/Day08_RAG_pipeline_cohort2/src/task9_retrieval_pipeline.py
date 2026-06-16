"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Điểm ngưỡng tối thiểu, nếu thấp hơn sẽ chuyển hướng tìm kiếm PageIndex
SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "mmr"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    dense_only: bool = False,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse (nếu không chạy dense_only)
          │
          ├→ Merge (RRF) → merged_results (hoặc bỏ qua nếu dense_only)
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không
        dense_only: Chỉ tìm kiếm ngữ nghĩa (dense), không dùng sparse và reranking

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    print(f"\n[Retrieval] Đang tìm kiếm cho truy vấn: '{query}' (dense_only={dense_only})")

    if dense_only:
        # Step 1 (dense only): Chỉ tìm kiếm ngữ nghĩa
        dense_results = semantic_search(query, top_k=top_k)
        for item in dense_results:
            item["source"] = "hybrid"
        final_results = dense_results
    else:
        # Step 1: Chạy tìm kiếm ngữ nghĩa (Dense) và từ khóa (Sparse)
        # Ta lấy gấp đôi số lượng top_k để có tập ứng viên phong phú cho bước reranking
        dense_results = semantic_search(query, top_k=top_k * 2)
        sparse_results = lexical_search(query, top_k=top_k * 2)

        # Lưu lại map điểm số ngữ nghĩa gốc để hiển thị chính xác và kiểm tra ngưỡng threshold
        dense_scores = {item["content"]: item["score"] for item in dense_results}

        # Step 2: Merge bằng RRF (Reciprocal Rank Fusion)
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
        for item in merged:
            item["source"] = "hybrid"

        # Step 3: Tiến hành Rerank để sắp xếp lại độ liên quan chính xác nhất
        if use_reranking and merged:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        else:
            final_results = merged[:top_k]

        # Khôi phục điểm số ngữ nghĩa gốc (nếu có) để hiển thị đẹp trên UI và phục vụ việc so khớp ngưỡng threshold
        for item in final_results:
            if item["content"] in dense_scores:
                item["score"] = dense_scores[item["content"]]

    # Step 4: Kiểm tra điểm số cao nhất với ngưỡng (threshold)
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(f"  ⚠ Điểm số Hybrid tốt nhất ({best_score:.3f}) thấp hơn ngưỡng tối thiểu ({score_threshold}).")
        print("  🔄 Đang kích hoạt chế độ dự phòng (Fallback) sang PageIndex...")
        
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            print(f"✓ Lấy kết quả thành công từ PageIndex ({len(fallback)} kết quả).")
            return fallback
        else:
            print("  ⚠ PageIndex không trả về kết quả (hoặc chưa cấu hình API key). Giữ lại kết quả Hybrid.")

    print(f"✓ Trả về {len(final_results)} kết quả Hybrid chất lượng tốt nhất.")
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
