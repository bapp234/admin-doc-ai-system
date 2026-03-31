# ĐẶC TẢ KIẾN TRÚC VÀ THUẬT TOÁN - MODULE NLP
**Dự án:** Admin Doc AI System  
**Phân hệ:** Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)

---

## 1. Tổng quan Module NLP
Sau quá trình số hóa hình ảnh, Module NLP đảm nhận vai trò phân tích ngữ nghĩa văn bản. Phân hệ này được thiết kế theo kiến trúc "Cắm và Chạy" (Plug-and-Play Interfaces), cho phép khởi động bằng các bộ quy tắc Heuristic (Rule-based) trong giai đoạn đầu và chuyển đổi mượt mà sang các mô hình Deep Learning/Transformer khi hệ thống được huấn luyện hoàn chỉnh.

## 2. Tiền xử lý và Bootstrapping Dữ liệu (`build_dataset.py`)
Giải quyết bài toán "Khởi tạo dữ liệu lạnh" (Cold-start Data Problem) cho quá trình huấn luyện máy học.
* **Thuật toán Tự động gán nhãn (Auto-labeling Heuristic):** Sử dụng phương pháp Keyword Matching để tự động phân lớp các văn bản đầu ra từ OCR thành tập dữ liệu huấn luyện hạt giống (Seed Dataset).
* Các quy tắc Mapping ánh xạ tập nhãn mục tiêu $Y = \{\text{QUYET\_DINH}, \text{THONG\_BAO}, \text{BAO\_CAO}, \text{CONG\_VAN}, \text{BANG\_CAP}\}$ dựa trên các mẫu chuỗi con (substring patterns) đặc trưng của văn bản hành chính Việt Nam.

## 3. Phân loại Văn bản - Text Classification (`classifier.py`)
Mô-đun này giải quyết bài toán cốt lõi của **Luận văn 2**. [cite_start]Mục tiêu của đề tài là xây dựng hệ thống phân loại văn bản hành chính tiếng Việt sử dụng các mô hình ngôn ngữ hiện đại[cite: 66].
* **Kiến trúc Thiết kế:** Khai báo lớp `TextClassifier` hỗ trợ song song hai chế độ: Mô phỏng quy tắc (Mock Mode) và Suy luận Máy học (Inference Mode).
* [cite_start]**Mô hình Mục tiêu (Transformer):** Hệ thống được cấu trúc để tương thích với các mô hình ngôn ngữ lớn như PhoBERT, viBERT hoặc XLM-R[cite: 16]. [cite_start]Các mô hình này sẽ được huấn luyện bằng HuggingFace Transformers và PyTorch[cite: 73].
* **Cơ sở Toán học của Text Classification (Softmax Layer):** Đối với văn bản đầu vào $x$, mô hình Transformer trích xuất vector biểu diễn $h_{CLS}$. Xác suất văn bản thuộc lớp $y_i$ được tính bằng hàm Softmax:
  
  $$P(y_i | x) = \frac{\exp(W_i \cdot h_{CLS} + b_i)}{\sum_{j=1}^{K} \exp(W_j \cdot h_{CLS} + b_j)}$$
  
  *(Trong đó, $K$ là tổng số loại văn bản hành chính, $W$ và $b$ là ma trận trọng số học được từ dataset huấn luyện).*

## 4. Trích xuất Thực thể và Thông tin - Information Extraction (`extractor.py`)
Mô-đun này hiện thực hóa các yêu cầu kỹ thuật của **Luận văn 3**.
* [cite_start]**Kiến trúc Trích xuất Lai (Hybrid Extraction Pipeline):** Thiết kế pipeline trích xuất thông tin gồm NER, regex và rule-based extraction[cite: 90]. Sự kết hợp này bù trừ khuyết điểm cho nhau: Regex có tốc độ $O(N)$ cực nhanh cho các định dạng cố định (như cấu trúc Ngày tháng `DD/MM/YYYY`), trong khi NER giải quyết tính đa hình của Ngôn ngữ tự nhiên.
* **Quy tắc Biểu thức chính quy (Regex):** Khai thác các cấu trúc neo (Anchor points) trong văn bản hành chính. Ví dụ, để bóc tách Số hiệu văn bản, hệ thống định nghĩa Automata:
  `r'(Số:|Số hiệu|Số vào sổ)[^\w]*([A-Z0-9/\.\-]+)'`
* [cite_start]**Mô hình Mục tiêu (Named Entity Recognition - NER):** Kỹ thuật Named Entity Recognition (NER) trong NLP [cite: 88] được áp dụng thông qua cấu trúc mạng `PhoBERT + CRF` (Conditional Random Field). [cite_start]Sử dụng mô hình PhoBERT NER để nhận diện các thực thể như ngày tháng, số văn bản và tên tổ chức[cite: 91].
* **Cơ sở Toán học của CRF trong NER:** Trong bài toán gán nhãn chuỗi (Sequence Labeling), với chuỗi đầu vào $X = \{x_1, x_2, ..., x_n\}$ và chuỗi nhãn thực thể $Y = \{y_1, y_2, ..., y_n\}$, điểm số của một cấu hình nhãn được tính bằng:
  
  $$s(X, Y) = \sum_{i=1}^{n} P_{i, y_i} + \sum_{i=1}^{n+1} A_{y_{i-1}, y_i}$$
  
  *(Trong đó $P_{i, y_i}$ là ma trận điểm phát xạ từ PhoBERT, và $A$ là ma trận chuyển trạng thái giữa các nhãn, giúp mô hình hiểu rằng nhãn `I-PER` (Inside Person) phải đứng sau `B-PER` (Begin Person)).*