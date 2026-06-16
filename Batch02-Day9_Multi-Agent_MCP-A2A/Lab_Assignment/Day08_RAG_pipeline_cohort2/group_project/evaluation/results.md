# RAG Evaluation Results

## Framework sử dụng

> **DeepEval (Chế độ chạy Local / Đánh giá Local)**

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A - B) |
|---|---|---|---|
| Faithfulness | 0.874 | 0.690 | +0.184 |
| Answer Relevance | 0.907 | 0.723 | +0.184 |
| Context Recall | 0.857 | 0.608 | +0.249 |
| Context Precision | 0.901 | 0.655 | +0.246 |
| **Average** | **0.885** | **0.669** | **+0.216** |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + Reranking):**
- Sử dụng kết hợp Dense Search (Vector database) và Sparse Search (BM25 Lexical search).
- Áp dụng cơ chế Reciprocal Rank Fusion (RRF) để kết hợp kết quả.
- Sử dụng Reranker (MMR) để sắp xếp lại độ liên quan.
- Tránh lost-in-the-middle bằng cách sắp xếp lại thứ tự ngữ cảnh trước khi nạp vào LLM.

**Config B (Dense-only, Không Reranking):**
- Chỉ sử dụng Dense Search (Vector database) thuần túy.
- Không áp dụng Lexical search (BM25), không dùng Reranking.
- Đưa thẳng top k kết quả từ Vector DB vào prompt của LLM theo thứ tự điểm số gốc.

**Kết luận:**
> **Config A (Hybrid + Rerank)** mang lại hiệu quả vượt trội so với Config B (Dense-only) với điểm số trung bình tăng +0.216.
> Việc kết hợp BM25 giúp cải thiện rõ rệt khả năng thu hồi các thuật ngữ chính xác trong văn bản luật (Context Recall), trong khi Reranking giúp đẩy các phần thông tin đắt giá lên đầu (Context Precision), giúp câu trả lời của LLM chính xác và giảm thiểu tình trạng giả định/hư cấu (Faithfulness).

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|---|---|---|---|---|---|
| 1 | Hoạt chất ma túy nào được tiêu thụ nh... | 0.84 | 0.86 | 0.82 | Fine-tuning | Sai lệch nhỏ về cách trích dẫn điều khoản hoặc cấu trúc câu. |
| 2 | Hình phạt cho tội tàng trữ trái phép ... | 0.86 | 0.89 | 0.79 | Fine-tuning | Sai lệch nhỏ về cách trích dẫn điều khoản hoặc cấu trúc câu. |
| 3 | Thời hạn quản lý người sử dụng trái p... | 0.84 | 0.89 | 0.84 | Fine-tuning | Sai lệch nhỏ về cách trích dẫn điều khoản hoặc cấu trúc câu. |

---

## Recommendations

### Cải tiến 1
**Action:** Tối ưu hóa bộ lọc từ dừng và chuẩn hóa tiếng Việt trước khi đưa vào BM25 Sparse Search.
**Expected impact:** Nâng cao chất lượng tìm kiếm từ khóa, tăng Context Recall đặc biệt cho các câu hỏi chứa số hiệu điều luật cụ thể.

### Cải tiến 2
**Action:** Tích hợp bộ Reranking mạnh hơn như Cross-Encoder hoặc Cohere Rerank thay cho MMR.
**Expected impact:** Sắp xếp lại chính xác hơn độ liên quan ngữ nghĩa của các chunk pháp luật phức tạp, từ đó tối ưu hóa Context Precision.

### Cải tiến 3
**Action:** Tinh chỉnh System Prompt của LLM để kiểm soát chặt chẽ nhiệt độ (Temperature = 0.1) và tăng cường kiểm tra tính xác thực của Citation.
**Expected impact:** Hạn chế triệt để ảo giác thông tin, nâng cao Faithfulness đối với các tác vụ tư vấn luật nhạy cảm.
