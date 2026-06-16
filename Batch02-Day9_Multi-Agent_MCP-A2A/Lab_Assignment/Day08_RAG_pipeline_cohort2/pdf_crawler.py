import os
from markitdown import MarkItDown

md = MarkItDown()

# Định nghĩa thư mục đầu ra và tự động tạo nếu chưa có
output_dir = "/Users/nguyendonganh/Day08_RAG_pipeline_cohort2/data/standardized"
os.makedirs(output_dir, exist_ok=True)

# 1. Chuyển đổi file PDF
pdf_path = "/Users/nguyendonganh/Day08_RAG_pipeline_cohort2/data/landing/tai_lieu_ma_tuy.pdf"
if os.path.exists(pdf_path):
    result = md.convert(pdf_path)
    # SỬA: Chỉ định tên file cụ thể trong thư mục đích
    output_path = os.path.join(output_dir, "Tai_lieu_ma_tuy.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    print(f" Đã chuyển đổi và lưu PDF vào: {output_path}")
else:
    print(f"❌ Lỗi: Không tìm thấy file PDF tại: {pdf_path}")

# 2. Chuyển đổi file DOCX
docx_path = "/Users/nguyendonganh/Day08_RAG_pipeline_cohort2/data/landing/legal/12151_Luat_Phong_chong_ma_tuy.docx"
if os.path.exists(docx_path):
    result = md.convert(docx_path)
    # SỬA: Chỉ định tên file cụ thể trong thư mục đích để không bị ghi đè
    output_path = os.path.join(output_dir, "12151_Luat_Phong_chong_ma_tuy.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    print(f" Đã chuyển đổi và lưu DOCX vào: {output_path}")
else:
    print(f"❌ Lỗi: Không tìm thấy file DOCX tại: {docx_path}")