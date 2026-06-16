# Báo cáo Cá nhân (Individual Reflection Report)
**Họ và tên:** Nguyễn Đông Anh  
**Mã số học viên (MSHV):** 2A202600760  
**Vai trò trong nhóm:** AI Engineer / Tech Lead  

---

## 💻 1. Đóng góp Kỹ thuật (Engineering Contribution)

Trong dự án **AI Evaluation Factory**, tôi đã chịu trách nhiệm thiết kế và triển khai các thành phần cốt lõi của hệ thống:

1. **Local TF-IDF Retriever & Chunking Ingestion**:
   - Triển khai thuật toán tách chunk tài liệu chính sách `data/company_policy.txt` dựa trên các tag tiêu đề `[DOC_XXX_YYY]`.
   - Viết thuật toán tìm kiếm và xếp hạng tài liệu cục bộ dựa trên tần suất từ khóa TF-IDF. Hỗ trợ hai chế độ:
     - **V1 (Base)**: Trọng số tiêu đề cơ bản.
     - **V2 (Optimized)**: Tối ưu hóa trọng số tiêu đề (gấp 5 lần nội dung) đóng vai trò như một bộ Reranker đơn giản giúp tăng Hit Rate và MRR.
   - Việc tự xây dựng bộ Retriever cục bộ giúp hệ thống chạy nhanh hơn đáng kể, hoàn toàn bảo mật và không tốn phí API cho bước tìm kiếm.

2. **Multi-Judge Consensus Engine**:
   - Triển khai logic gọi đồng thời hai Judge độc lập: Giám khảo 1 (`gemini-3.5-flash`) và Giám khảo 2 (`gemini-2.5-flash`) thông qua cơ chế bất đồng bộ `asyncio.gather`.
   - Thiết kế thuật toán tính toán độ đồng thuận và **Conflict Resolution**: Nếu điểm số giữa 2 giám khảo lệch nhau > 1 điểm, hệ thống tự động kích hoạt Giám khảo trưởng (Master Judge dùng model `gemini-2.5-pro`) để hòa giải và đưa ra điểm số cuối cùng kèm lý do đồng thuận chi tiết.
   - Triển khai kiểm tra thiên vị vị trí (**Position Bias Check**) bằng cách tráo đổi thứ tự câu trả lời hiển thị cho Judge.

3. **Async Benchmark Runner & Cost Tracker**:
   - Tối ưu hóa hiệu năng chạy benchmark bằng `asyncio.Semaphore` để khống chế số lượng request đồng thời gửi lên API, hạn chế tối đa lỗi 429 (Rate Limit).
   - Triển khai bộ đếm Token chi tiết dựa trên dữ liệu sử dụng thực tế trả về từ Gemini API (nhập/xuất) và tính toán chi phí chi tiết theo USD cho từng test case.

---

## 📚 2. Chiều sâu Kỹ thuật (Technical Depth)

### 2.1. Mean Reciprocal Rank (MRR)
MRR là một chỉ số quan trọng để đánh giá chất lượng của giai đoạn Retrieval (Tìm kiếm thông tin). Thay vì chỉ kiểm tra xem tài liệu đúng có xuất hiện trong kết quả hay không (Hit Rate), MRR đánh giá xem tài liệu đúng đó xuất hiện ở **vị trí thứ mấy** trong danh sách kết quả trả về.
Công thức tính cho một truy vấn $q$:
$$RR(q) = \frac{1}{\text{rank}_i}$$
Trong đó $\text{rank}_i$ là vị trí đầu tiên của tài liệu liên quan trong danh sách kết quả (1-indexed). Nếu không tìm thấy tài liệu liên quan nào, $RR(q) = 0$.
MRR trung bình trên tập dữ liệu là trung bình cộng của các $RR(q)$. Chỉ số này nằm trong khoảng $[0, 1]$, càng gần 1 chứng tỏ hệ thống RAG tìm thấy tài liệu chuẩn ở các vị trí đầu tiên rất tốt, giúp LLM nhận được thông tin liên quan nhanh nhất và tránh bị nhiễu thông tin ở các chunk dưới.

### 2.2. Cohen's Kappa & Agreement Rate
Để đo lường độ tin cậy của Multi-Judge, ta sử dụng **Agreement Rate** (Tỷ lệ đồng thuận). Nó đo lường tần suất các Judge đưa ra quyết định tương đồng nhau (ở đây chúng tôi định nghĩa đồng thuận là điểm số lệch nhau $\le 1$ điểm trên thang 5).
Trong các hệ thống phức tạp hơn, hệ số **Cohen's Kappa** được sử dụng để loại bỏ yếu tố đồng thuận ngẫu nhiên:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
Trong đó $p_o$ là tỷ lệ đồng thuận quan sát được, và $p_e$ là tỷ lệ đồng thuận kỳ vọng ngẫu nhiên. Kappa $> 0.6$ biểu thị sự đồng thuận tốt giữa các Judge, đảm bảo hệ thống đánh giá là khách quan và không bị phụ thuộc vào một model riêng lẻ.

### 2.3. Position Bias (Thiên vị Vị trí)
Position Bias là một lỗi phổ biến của các LLM-as-a-Judge, khi mô hình có xu hướng chấm điểm cao hơn cho câu trả lời xuất hiện đầu tiên (hoặc xuất hiện cuối cùng) trong prompt so sánh, bất kể chất lượng thực tế.
Để khắc phục Position Bias:
1. **Tráo đổi vị trí (Order Swapping)**: Gửi 2 request riêng biệt đến Judge: lần 1 đưa Answer A lên trước, lần 2 đưa Answer B lên trước.
2. **Đối chiếu kết quả**: Nếu Judge chọn Answer A khi A đứng trước, nhưng lại chọn Answer B khi B đứng trước, hệ thống sẽ phát hiện hành vi "Position Bias" và tiến hành tính trung bình điểm hoặc gọi Master Judge xử lý.

### 2.4. Trade-off giữa Chi phí và Chất lượng trong Evaluation
Việc đánh giá hệ thống AI bằng LLM (LLM-as-a-Judge) mang lại độ chính xác cao nhưng chi phí rất đắt đỏ và độ trễ cao. Để tối ưu hóa, tôi đề xuất các giải pháp:
- **Tối ưu hóa Judge Selection**: Sử dụng model Flash rẻ tiền (`gemini-3.5-flash`, `gemini-2.5-flash`) cho 90% các đánh giá cơ bản. Chỉ gọi model Pro đắt tiền (`gemini-2.5-pro`) để giải quyết xung đột khi điểm số giữa các Judge Flash lệch nhau đáng kể. Phương án này giúp giảm trên **30% chi phí** so với việc dùng toàn bộ model Pro mà vẫn giữ nguyên độ chính xác tương đương.
- **Caching**: Lưu trữ kết quả đánh giá đối với các câu trả lời trùng lặp hoặc không thay đổi qua các phiên bản để tiết kiệm API token.

---

## 🛠️ 3. Giải quyết Vấn đề (Problem Solving)

Trong quá trình thực hiện Lab Day 14, tôi đã gặp và giải quyết các vấn đề lớn sau:

1. **Lỗi Quota API Exceeded (HTTP 429) của tài khoản Free**:
   - *Vấn đề*: Tài khoản API Gemini miễn phí bị giới hạn ngặt nghèo ở mức 5 RPM (Requests Per Minute). Khi sinh dữ liệu (SDG) hoặc chạy song song 50 test cases, hệ thống lập tức sập do lỗi 429.
   - *Giải pháp*: 
     - Thiết kế lại script sinh dữ liệu chạy tuần tự và có thời gian nghỉ `time.sleep(15)` bắt buộc giữa các yêu cầu để đảm bảo tần suất gửi dưới 4 RPM.
     - Triển khai cơ chế tự động thử lại (Retry Loop) với thời gian chờ tăng dần (exponential backoff) nếu phát hiện lỗi 429 hoặc lỗi quota quá hạn.
     - Thiết lập cơ chế chạy song song có kiểm soát bằng `asyncio.Semaphore(1)` và trì hoãn nhỏ giữa các case trong Runner để chạy benchmark ổn định tuyệt đối mà không bị ngắt quãng.

2. **Lỗi Treo (Hang) và Cảnh báo Thư viện Google-GenerativeAI**:
   - *Vấn đề*: Thư viện gRPC của `google-generativeai` hoạt động không ổn định trên môi trường macOS khi chạy bất đồng bộ, gây ra hiện tượng treo thread và cảnh báo không hỗ trợ phiên bản Python 3.9 cũ của hệ thống.
   - *Giải pháp*: Tôi đã loại bỏ hoàn toàn việc sử dụng SDK `google-generativeai` và chuyển sang giao tiếp trực tiếp bằng **Gemini REST API** thông qua thư viện `requests` tiêu chuẩn. Giải pháp này giúp loại bỏ hoàn toàn các cảnh báo thư viện lỗi thời, kiểm soát timeout 30 giây chặt chẽ và hoạt động mượt mà 100%.

3. **Mô phỏng Retrieval Độc lập**:
   - *Vấn đề*: Sinh viên cần đo đạc Hit Rate và MRR cho Retrieval stage nhưng template không tích hợp sẵn Vector DB thực tế.
   - *Giải pháp*: Triển khai một công cụ tìm kiếm cục bộ TF-IDF hoàn chỉnh ngay trong file `agent/main_agent.py`. Giúp mô phỏng chính xác hành vi của Vector DB, trả về danh sách các chunk ID khớp nhất để tính toán các chỉ số Hit Rate và MRR một cách trực quan và thực tế nhất.
