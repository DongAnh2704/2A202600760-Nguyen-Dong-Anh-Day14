"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import os
import pickle
from pathlib import Path
import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# CHUNK_SIZE = 500: Chọn kích thước vừa đủ để giữ trọn vẹn ngữ nghĩa của một điều khoản luật
# CHUNK_OVERLAP = 50: Giúp liên kết ngữ nghĩa giữa các chunk liền kề, tránh mất thông tin ở ranh giới cắt
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# BAAI/bge-m3: Mô hình embedding đa ngôn ngữ cực tốt cho Tiếng Việt, có khả năng dense/sparse/colbert
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Chọn FAISS làm Vector Store chính vì không chạy Weaviate trên máy local
VECTOR_STORE = "faiss"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"⚠ Thư mục {STANDARDIZED_DIR} không tồn tại!")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.name.startswith('.'):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            # Xác định loại tài liệu dựa trên đường dẫn
            doc_type = "legal" if "legal" in md_file.parts else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
        except Exception as e:
            print(f"❌ Lỗi đọc file {md_file.name}: {e}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            cleaned_text = chunk_text.strip()
            # Loại bỏ các chunk quá ngắn dưới 80 ký tự (ví dụ: số trang, thanh kẻ bảng biểu |, tiêu đề con rời rạc)
            if len(cleaned_text) >= 80:
                chunks.append({
                    "content": cleaned_text,
                    "metadata": {
                        "source": doc["metadata"]["source"],
                        "type": doc["metadata"]["type"],
                        "chunk_index": i
                    }
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"Đang tải mô hình embedding: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [c["content"] for c in chunks]
    print(f"Đang tạo vector embedding cho {len(chunks)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_weaviate(chunks: list[dict]):
    """Lưu chunks vào Weaviate local."""
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    print("Đang kết nối tới Weaviate local...")
    with weaviate.connect_to_local() as client:
        if client.collections.exists("DrugLawDocs"):
            print("  Xóa Collection cũ...")
            client.collections.delete("DrugLawDocs")

        print("  Khởi tạo Collection 'DrugLawDocs'...")
        collection = client.collections.create(
            name="DrugLawDocs",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )

        print("  Đang ghi dữ liệu (Batch Insert)...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                    vector=chunk["embedding"]
                )
    print("✓ Đã hoàn tất đánh chỉ mục vào Weaviate!")


def index_to_chromadb(chunks: list[dict]):
    """Lưu chunks vào ChromaDB local."""
    import chromadb

    print("Đang kết nối tới ChromaDB local...")
    chroma_client = chromadb.PersistentClient(path="data/chroma")
    
    try:
        chroma_client.delete_collection("DrugLawDocs")
    except Exception:
        pass
        
    collection = chroma_client.create_collection("DrugLawDocs")

    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("  Đang lưu chunks vào ChromaDB...")
    for start_idx in range(0, len(chunks), 4000):
        end_idx = start_idx + 4000
        collection.add(
            documents=documents[start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            ids=ids[start_idx:end_idx]
        )
    print("✓ Đã hoàn tất đánh chỉ mục vào ChromaDB!")


def index_to_faiss(chunks: list[dict]):
    """Lưu chunks vào FAISS index local."""
    import faiss

    print("Đang khởi tạo index FAISS...")
    os.makedirs("data/faiss", exist_ok=True)

    emb_matrix = np.array([c["embedding"] for c in chunks], dtype="float32")
    dim = emb_matrix.shape[1]

    # Chuẩn hóa L2 để tìm kiếm cosine similarity bằng Inner Product
    faiss.normalize_L2(emb_matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)

    faiss.write_index(index, "data/faiss/index.faiss")

    # Lưu metadata (không lưu kèm vector để tiết kiệm bộ nhớ)
    chunks_to_save = []
    for c in chunks:
        chunks_to_save.append({
            "content": c["content"],
            "metadata": c["metadata"]
        })
    
    with open("data/faiss/chunks.pkl", "wb") as f:
        pickle.dump(chunks_to_save, f)
        
    print("✓ Đã hoàn tất lưu index và metadata sang FAISS!")


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào vector store đã cấu hình, tự động dự phòng nếu lỗi."""
    global VECTOR_STORE
    if not chunks:
        print("⚠ Không có chunk nào để đánh chỉ mục.")
        return

    try:
        if VECTOR_STORE == "weaviate":
            index_to_weaviate(chunks)
        elif VECTOR_STORE == "chromadb":
            index_to_chromadb(chunks)
        elif VECTOR_STORE == "faiss":
            index_to_faiss(chunks)
    except Exception as e:
        print(f"❌ Gặp lỗi khi lưu vào Vector Store '{VECTOR_STORE}': {e}")
        # Cơ chế dự phòng thông minh
        if VECTOR_STORE != "faiss":
            print("\n🔄 Đang thực hiện phương án DỰ PHÒNG: Đánh chỉ mục sang FAISS (Local)...")
            try:
                index_to_faiss(chunks)
                # Thay đổi cấu hình toàn cục sang faiss cho các module sau sử dụng
                VECTOR_STORE = "faiss"
                print("✓ Đã lưu dự phòng FAISS thành công!")
            except Exception as fe:
                print(f"❌ Lỗi nghiêm trọng: Không thể lưu dự phòng FAISS: {fe}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
