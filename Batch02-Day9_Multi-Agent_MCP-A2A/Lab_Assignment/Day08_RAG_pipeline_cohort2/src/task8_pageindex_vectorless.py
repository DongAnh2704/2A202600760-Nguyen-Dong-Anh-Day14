"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_xxx"):
        print("⚠ Bỏ qua bước upload: Không tìm thấy PAGEINDEX_API_KEY hợp lệ trong file .env")
        return

    try:
        from pageindex import PageIndex
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)

        if not STANDARDIZED_DIR.exists():
            print(f"⚠ Thư mục {STANDARDIZED_DIR} không tồn tại.")
            return

        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            if md_file.name.startswith('.'):
                continue
            content = md_file.read_text(encoding="utf-8")
            pi.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name}
            )
            print(f"  ✓ Uploaded: {md_file.name}")
        print("✓ Đã hoàn tất upload toàn bộ tài liệu lên PageIndex.")
    except Exception as e:
        print(f"❌ Lỗi khi tải tài liệu lên PageIndex: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_xxx"):
        print("⚠ Bỏ qua tìm kiếm PageIndex: Không tìm thấy PAGEINDEX_API_KEY hợp lệ.")
        return []

    try:
        from pageindex import PageIndex
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)

        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": r.metadata,
                "source": "pageindex"
            }
            for r in results
        ]
    except Exception as e:
        print(f"❌ Lỗi khi truy vấn PageIndex: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_xxx"):
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
