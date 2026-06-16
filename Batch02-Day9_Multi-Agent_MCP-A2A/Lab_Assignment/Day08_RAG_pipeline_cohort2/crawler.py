import os
import re
import asyncio
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler

async def crawl_article(url: str, output_dir: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        # 1. Đảm bảo thư mục output tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Tạo tên file hợp lệ từ URL (loại bỏ các ký tự đặc biệt)
        parsed_url = urlparse(url)
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', parsed_url.netloc + parsed_url.path)
        if safe_filename.endswith('_'):
            safe_filename = safe_filename[:-1]
        filename = f"{safe_filename or 'extracted_content'}.md"
        file_path = os.path.join(output_dir, filename)
        
        # 3. Lưu result.markdown vào file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)
            
        print(f" Đã lưu nội dung vào: {file_path}")

if __name__ == "__main__":
    target_url = "https://bvhttdl.gov.vn/bac-ninh-tuyen-truyen-phong-chong-ma-tuy-trong-cac-hoat-dong-van-hoa-van-nghe-the-duc-the-thao-du-lich-20260202200530837.htm"
    output_directory = "data"
    asyncio.run(crawl_article(target_url, output_directory))