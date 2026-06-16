"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://congan.tuyenquang.gov.vn/vi/tin-bai/noi-khong-voi-ma-tuy-bao-ve-tuong-lai-cua-chinh-minh?type=NEWS&id=233701",
    "https://bvhttdl.gov.vn/bac-ninh-tuyen-truyen-phong-chong-ma-tuy-trong-cac-hoat-dong-van-hoa-van-nghe-the-duc-the-thao-du-lich-20260202200530837.htm",
    "https://namphu.hanoi.gov.vn/an-ninh-quoc-phong-64818/bai-tuyen-truyen-tac-hai-cua-ma-tuy-va-cach-phong-chong-2721260605115523797.htm",
    "https://tiengchuong.chinhphu.vn/canh-bao-hiem-hoa-ma-tuy-ngay-gia-tang-trong-gioi-tre-113251124131941924.htm",
    "https://khoahocphaplyvietnam.edu.vn/khplvn/article/view/357"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        # Lấy tiêu đề từ metadata hoặc trích xuất từ URL
        title = None
        if result.metadata:
            title = result.metadata.get("title") or result.metadata.get("og:title")
        
        if not title:
            domain = url.split('/')[2]
            title = f"Bài viết từ {domain}"
            
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)

            # Lưu file JSON
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ Saved: {filepath}")
        except Exception as e:
            print(f"❌ Lỗi khi crawl URL {url}: {e}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
