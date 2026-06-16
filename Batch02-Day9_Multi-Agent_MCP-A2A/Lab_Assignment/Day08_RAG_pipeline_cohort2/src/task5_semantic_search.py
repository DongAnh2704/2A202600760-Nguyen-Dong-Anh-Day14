"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
from sentence_transformers import SentenceTransformer

# Nhập cấu hình từ Task 4, sử dụng fallback nếu chạy trực tiếp file này
try:
    from .task4_chunking_indexing import EMBEDDING_MODEL, VECTOR_STORE
except ImportError:
    EMBEDDING_MODEL = "BAAI/bge-m3"
    VECTOR_STORE = "faiss"

# Khởi tạo mô hình Embedding (lazy load để tối ưu hiệu năng)
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        # Tải mô hình đã cấu hình ở Task 4 (ví dụ: BAAI/bge-m3)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Bước 1: Sinh vector embedding cho câu truy vấn
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    search_results = []

    # Hàm con phục vụ việc tìm kiếm dự phòng bằng FAISS
    def run_faiss_fallback():
        nonlocal search_results
        import faiss
        import pickle
        import numpy as np

        index_path = "data/faiss/index.faiss"
        chunks_path = "data/faiss/chunks.pkl"

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            try:
                index = faiss.read_index(index_path)
                with open(chunks_path, "rb") as f:
                    chunks = pickle.load(f)
                
                # Chuẩn hóa vector truy vấn và tìm kiếm
                xq = np.array([query_embedding], dtype="float32")
                faiss.normalize_L2(xq)
                distances, indices = index.search(xq, top_k)
                
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1 and idx < len(chunks):
                        search_results.append({
                            "content": chunks[idx]["content"],
                            "score": float(dist),
                            "metadata": chunks[idx]["metadata"]
                        })
                print("✓ Đã tìm kiếm thành công từ nguồn dữ liệu dự phòng FAISS (Local).")
            except Exception as fe:
                print(f"❌ Lỗi khi truy vấn file index FAISS: {fe}")
        else:
            print(f"⚠️ Không tìm thấy file index FAISS tại {index_path} hoặc {chunks_path} để dự phòng.")

    # Bước 2: Tìm kiếm trên Vector Store tương ứng
    if VECTOR_STORE == "weaviate":
        import weaviate
        from weaviate.classes.query import MetadataQuery

        # Kết nối tới instance Weaviate local
        try:
            with weaviate.connect_to_local() as client:
                collection = client.collections.get("DrugLawDocs")
                results = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=top_k,
                    return_metadata=MetadataQuery(distance=True)
                )
                
                for obj in results.objects:
                    # Cosine similarity = 1 - cosine_distance.
                    distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
                    score = 1.0 - distance
                    search_results.append({
                        "content": obj.properties.get("content", ""),
                        "score": float(score),
                        "metadata": {
                            "source": obj.properties.get("source", ""),
                            "doc_type": obj.properties.get("doc_type", ""),
                            "chunk_index": int(obj.properties.get("chunk_index", 0))
                        }
                    })
        except Exception as e:
            print(f"❌ Lỗi kết nối hoặc truy vấn Weaviate: {e}")
            print("🔄 Đang kích hoạt tìm kiếm dự phòng bằng FAISS...")
            run_faiss_fallback()

    elif VECTOR_STORE == "chromadb":
        import chromadb
        
        try:
            # Kết nối tới ChromaDB local
            chroma_client = chromadb.PersistentClient(path="data/chroma")
            collection = chroma_client.get_collection(name="DrugLawDocs")
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if results and results["documents"] and len(results["documents"]) > 0:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0] if "distances" in results else [0.0] * len(documents)
                
                for doc, meta, dist in zip(documents, metadatas, distances):
                    # ChromaDB cosine distance: similarity = 1 - distance
                    score = 1.0 - dist
                    search_results.append({
                        "content": doc,
                        "score": float(score),
                        "metadata": meta
                    })
        except Exception as e:
            print(f"❌ Lỗi kết nối hoặc truy vấn ChromaDB: {e}")
            print("🔄 Đang kích hoạt tìm kiếm dự phòng bằng FAISS...")
            run_faiss_fallback()

    elif VECTOR_STORE == "faiss":
        run_faiss_fallback()

    # Sắp xếp lại danh sách kết quả theo độ tương đồng giảm dần
    search_results = sorted(search_results, key=lambda x: x["score"], reverse=True)
    return search_results


if __name__ == "__main__":
    # Chạy thử nghiệm tìm kiếm
    test_query = "hình phạt cho tội tàng trữ ma tuý"
    print(f"Đang thực hiện Semantic Search cho câu hỏi: '{test_query}'...")
    results = semantic_search(test_query, top_k=5)
    
    if results:
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    else:
        print("Không tìm thấy kết quả hoặc lỗi kết nối Vector Store.")
