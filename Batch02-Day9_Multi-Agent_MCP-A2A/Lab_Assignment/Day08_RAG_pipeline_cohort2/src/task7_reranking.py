"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
import requests
import numpy as np
from typing import Optional

# Khởi tạo mô hình CrossEncoder (lazy load)
_reranker_model = None

def get_reranker_model():
    """Tải và khởi tạo mô hình CrossEncoder cục bộ."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        # Sử dụng mô hình BAAI/bge-reranker-base hỗ trợ tiếng Việt tốt
        _reranker_model = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker_model


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    jina_api_key = os.getenv("JINA_API_KEY", "")
    
    # 1. Nếu có Jina API Key hợp lệ, gọi API
    if jina_api_key and not jina_api_key.startswith("jina_xxx"):
        try:
            print("Đang gọi Jina Reranker API...")
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json()["results"]
                results = []
                for r in reranked:
                    idx = r["index"]
                    item = candidates[idx].copy()
                    item["score"] = float(r["relevance_score"])
                    results.append(item)
                return results
        except Exception as e:
            print(f"⚠️ Lỗi gọi Jina API: {e}. Chuyển sang mô hình cục bộ...")

    # 2. Phương án dự phòng: Sử dụng mô hình CrossEncoder cục bộ
    try:
        model = get_reranker_model()
        pairs = [[query, c["content"]] for c in candidates]
        scores = model.predict(pairs)
        
        # Tạo bản sao và cập nhật điểm số
        reranked_candidates = []
        for c, raw_score in zip(candidates, scores):
            item = c.copy()
            # Áp dụng hàm sigmoid để chuyển điểm sang khoảng 0-1
            item["score"] = float(1.0 / (1.0 + np.exp(-raw_score)))
            reranked_candidates.append(item)
            
        # Sắp xếp
        reranked_candidates = sorted(reranked_candidates, key=lambda x: x["score"], reverse=True)
        return reranked_candidates[:top_k]
    except Exception as e:
        print(f"⚠️ Không thể chạy Cross-Encoder cục bộ: {e}. Sắp xếp theo score gốc làm dự phòng.")
        # Dự phòng cơ bản nhất: giữ nguyên điểm số cũ
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return sorted_candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    # Định nghĩa hàm tính Cosine Similarity
    def cosine_sim(a, b):
        vec_a = np.array(a)
        vec_b = np.array(b)
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # Đảm bảo các candidates đều có vector embedding (nếu thiếu thì sinh ra)
    for c in candidates:
        if "embedding" not in c:
            try:
                from .task5_semantic_search import get_embedding_model
                emb_model = get_embedding_model()
                c["embedding"] = emb_model.encode(c["content"]).tolist()
            except Exception:
                c["embedding"] = [0.0] * len(query_embedding)

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance với query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Độ tương đồng lớn nhất với các tài liệu đã được chọn
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # Công thức MMR score
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            # Tính điểm RRF cộng dồn qua các bảng xếp hạng khác nhau
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sắp xếp các đoạn văn theo điểm RRF giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)  # cập nhật điểm số mới bằng điểm RRF
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking ("cross_encoder" | "mmr" | "rrf")

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        try:
            from .task5_semantic_search import get_embedding_model
            emb_model = get_embedding_model()
            query_embedding = emb_model.encode(query).tolist()
            return rerank_mmr(query_embedding, candidates, top_k)
        except Exception as e:
            print(f"⚠️ MMR rerank thất bại do không sinh được embedding cho query: {e}")
            return candidates[:top_k]
    elif method == "rrf":
        # Nếu truyền danh sách đơn vào RRF, ta coi đó là danh sách xếp hạng duy nhất
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    print("Đang chạy thử nghiệm Rerank...")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
