# Tasks

- [x] **Phần 1: Direct LLM Calling**
  - [x] Bài Tập 1.1: Sửa câu hỏi sang tiếng Việt trong stages/stage_1_direct_llm/main.py
  - [x] Bài Tập 1.2: Thêm custom temperature parameter cho get_llm trong common/llm.py
  - [x] Chạy và xác thực Stage 1

- [x] **Phần 2: LLM + RAG & Tools**
  - [x] Bài Tập 2.1: Thêm labor law entry vào LEGAL_KNOWLEDGE trong exercises/exercise_2_tools.py
  - [x] Bài Tập 2.2: Tạo check_statute_of_limitations tool, bind và xử lý trong exercises/exercise_2_tools.py
  - [x] Chạy và xác thực Exercise 2

- [x] **Phần 3: Single Agent với ReAct**
  - [x] Bài Tập 3.1: Tạo search_case_law tool, bind và đổi câu hỏi trong stages/stage_3_single_agent/main.py
  - [x] Bài Tập 3.2: Thêm verbose=True / debug logs cho Stage 3
  - [x] Chạy và xác thực Stage 3

- [x] **Phần 4: Multi-Agent In-Process**
  - [x] Bài Tập 4.1: Thêm privacy_agent & privacy_analysis field vào exercises/exercise_4_multiagent.py
  - [x] Bài Tập 4.2: Cập nhật conditional routing & build_graph trong exercises/exercise_4_multiagent.py
  - [x] Chạy và xác thực Exercise 4

- [x] **Phần 5: Distributed A2A System & Latency Optimization**
  - [x] Chạy start_all.sh & test_client.py ban đầu, đo latency gốc
  - [x] Phân tích request flow qua trace_id & test dynamic discovery bằng cách dừng Tax Agent
  - [x] Bài Tập 5.3: Sửa tax_agent prompt ngắn gọn hơn
  - [x] Tối ưu hóa: Implement caching cho registry_client.py và a2a_client.py
  - [x] Đo lại latency sau khi tối ưu hóa và tổng kết
