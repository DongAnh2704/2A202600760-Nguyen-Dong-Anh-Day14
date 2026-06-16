# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

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

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
# KIẾN TRÚC RAG PIPELINE

+-------------------------------------------------------------------------+
|                  GIAI ĐOẠN 1: OFFLINE DATA INGESTION                    |
+-------------------------------------------------------------------------+
|                                                                         |
|  [ Tài liệu thô ] (PDF, DOCX, TXT, Web Page, Markdown,...)              |
|         │                                                               |
|         ▼                                                               |
|  [ Document Loader ] (Trích xuất toàn bộ văn bản thô từ file)           |
|         │                                                               |
|         ▼                                                               |
|  [ Text Splitter ] (Cắt nhỏ văn bản thành các Chunk: size 512-1000)      |
|         │                                                               |
|         ▼                                                               |
|  [ Embedding Model ] (Chuyển đổi từng ký tự/Chunk thành Vector số)      |
|         │                                                               |
|         ▼                                                               |
|  [ Vector Database ] (Lưu trữ Vector kèm Metadata vào ChromaDB/FAISS)  |
|                                                                         |
+-------------------------------------------------------------------------+

===========================================================================

+-------------------------------------------------------------------------+
|                GIAI ĐOẠN 2: ONLINE RETRIEVAL & GENERATION               |
+-------------------------------------------------------------------------+
|                                                                         |
|   Người dùng nhập: [ Câu hỏi / Query ]                                   |
|                          │                                              |
|                          ▼                                              |
|                  [ Embedding Model ]                                    |
|             (Chuyển Câu hỏi thành Vector số)                            |
|                          │                                              |
|                          ▼                                              |
|       🔍 KẾT NỐI VỚI [ Vector Database ]                                |
|             (Tính toán tương đồng: Cosine Similarity / L2)              |
|                          │                                              |
|                          ├──► [ Tìm kiếm Top-K Chunks có liên quan nhất] |
|                          │                                              |
|                          ▼                                              |
|                  [ Prompt Template ]                                    |
|       (Đóng gói: Prompt hệ thống + Bộ ngữ cảnh Context + Câu hỏi)       |
|                          │                                              |
|                          ▼                                              |
|             🤖 [ Large Language Model ] (LLM)                            |
|         (Đọc Prompt, tổng hợp thông tin, chống ảo tưởng)                 |
|                          │                                              |
|                          ▼                                              |
|     Kết quả trả về: [ Câu trả lời hoàn chỉnh / Answer ]                 |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Lưu Công Thái | 2A202600949 | Leader/Quản lý Khung xương Pipeline | Hoạt động |
| Trảo An Huy | 2A202600819 | Quản lý Cơ sở dữ liệu Vector (Vector DB)/ Tối ưu Truy vấn | Hoạt động |
| Nguyễn Mạnh Đức | 2A202600734 | Đánh giá & Tối ưu hóa RAG (Evaluation & Optimization) | Hoạt động |
| Nguyễn Đông Anh | 2A202600760 | Kiểm Prompt & Quản lý Mô hình Ngôn ngữ lớn (LLM) | Hoạt động |
| Lê Quang Đạt | 2A202600630 | Đóng gói Giao diện (UI) | Hoạt động |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.