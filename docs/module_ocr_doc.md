# ĐẶC TẢ KIẾN TRÚC VÀ THUẬT TOÁN - MODULE OCR
**Dự án:** Admin Doc AI System  
**Phân hệ:** Tiền xử lý & Nhận dạng quang học (Core AI Engine)

---

## 1. Tổng quan Module OCR (OCR Pipeline)
Module OCR chịu trách nhiệm trích xuất văn bản từ hình ảnh tài liệu thô, duy trì cấu trúc không gian (bố cục 2D cột/hàng) và lọc bỏ nhiễu trước khi bàn giao cho Module NLP. Hệ thống sử dụng kiến trúc lai (Hybrid Architecture) kết hợp sức mạnh của nhiều mô hình Deep Learning SOTA (State-of-the-Art) và các thuật toán xử lý hình học máy tính.

## 2. Phân tích Bố cục (`layout.py`)
Module này chịu trách nhiệm định vị các vùng chứa chữ và phân loại cấu trúc văn bản để chống hiện tượng đọc lộn cột.

* **Core Model:** Sử dụng thư viện `docTR` với kiến trúc phát hiện (Detection) là **DBNet** (`db_resnet50`) và nhận dạng (`crnn_vgg16_bn`).
* **Thuật toán cốt lõi (Differentiable Binarization - DBNet):** Khác với binarization truyền thống (ngưỡng cứng), DBNet sử dụng một mạng nơ-ron để dự đoán bản đồ ngưỡng (threshold map) $T$ và bản đồ xác suất (probability map) $P$. Hàm binarization xấp xỉ được định nghĩa là:
  
  $$\hat{B}_{i,j} = \frac{1}{1 + e^{-k(P_{i,j} - T_{i,j})}}$$
  
  *(Trong đó, $k$ là hệ số phóng đại, thường $k=50$). Thuật toán này giúp mô hình thích ứng rất tốt với các văn bản hành chính bị nhiễu nền, mờ hoặc ố vàng.*
* **Quy trình Ánh xạ Tọa độ (Coordinate Mapping):** Tọa độ chuẩn hóa ban đầu thuộc khoảng $[0, 1]$. Để ánh xạ về kích thước ảnh thực tế $(W, H)$, hệ thống áp dụng phép nhân ma trận cơ bản:
  
  $$X_{pixel} = \lfloor x_{norm} \times W \rfloor, \quad Y_{pixel} = \lfloor y_{norm} \times H \rfloor$$
  
* **Reading Order Extraction:** Phân rã văn bản theo cấp bậc đồ thị có hướng `Page -> Block -> Line` để bảo toàn tính ngữ nghĩa theo thứ tự đọc tự nhiên của con người.

## 3. Nhận diện Chữ (`recognizer.py`)
Đảm nhiệm việc dịch mã các mảnh ảnh (cropped images) thành văn bản thuần túy.

* **Core Model:** `VietOCR` với kiến trúc mạng **VGG + Transformer**.
* **Cơ sở thuật toán:**
  1. **Trích xuất đặc trưng (Feature Extraction):** Mạng CNN (VGG) trích xuất ma trận đặc trưng không gian $X$ từ ảnh đầu vào.
  2. **Giải mã chuỗi (Sequence Decoding):** Kiến trúc Transformer sử dụng cơ chế **Self-Attention** để nắm bắt sự phụ thuộc ngữ cảnh dài (ví dụ: các dấu câu và dấu thanh trong tiếng Việt). Công thức Attention:
     
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
     
* **Tối ưu Suy luận (Inference Optimization):** Hệ thống chủ động tắt tính năng `beamsearch` (tìm kiếm chùm) và sử dụng thuật toán **Greedy Search**. Thuật toán Greedy chọn token có xác suất cao nhất tại mỗi bước thời gian $t$:
  
  $$\hat{y}_t = \arg\max_{y} P(y_t | y_1, ..., y_{t-1}, X)$$
  
  *(Giải pháp này giúp giảm 30-40% thời gian xử lý mà vẫn duy trì độ chính xác (CER) vượt trội đối với văn bản in).*

## 4. Xử lý Hình học (`cropper.py`)
Sử dụng hình học giải tích (Analytic Geometry) để nắn phẳng (Deskew) và tiền xử lý các mảnh ảnh chữ nhật chứa text.

* **Thuật toán Khôi phục Phối cảnh (Perspective Transform):**
  1. Tính toán kích thước mảnh ảnh mục tiêu dựa trên khoảng cách Euclidean (Norm bậc 2) lớn nhất giữa các đỉnh:
     
     $$Width = \max\left(\|P_0 - P_1\|_2, \|P_2 - P_3\|_2\right)$$
     $$Height = \max\left(\|P_0 - P_3\|_2, \|P_1 - P_2\|_2\right)$$
     
  2. Tìm ma trận biến đổi phối cảnh $M \in \mathbb{R}^{3 \times 3}$ sao cho nó ánh xạ các điểm gốc $P_{src}$ sang các điểm ảnh chữ nhật đích $P_{dst}$:
     
     $$P_{dst} = M \cdot P_{src}$$
     
  3. Áp dụng biến đổi để nội suy điểm ảnh.
* **Thuật toán Chống lật (Anti-flip Heuristic):** Đặc thù của tiếng Việt là chiều dài chuỗi chữ luôn lớn hơn chiều cao nét chữ. Nếu phát hiện dị thường $\frac{Height}{Width} > 1.2$, hệ thống tự động kích hoạt ma trận xoay affine góc $-90^\circ$ (ngược chiều kim đồng hồ) để nắn lại dòng chữ.
* **Padding Optimization:** Chủ động thêm viền trắng bằng thuật toán `BORDER_CONSTANT` để ngăn hiện tượng "mù biên" (Border Blindness) của mạng CNN ở các lớp Convolution đầu tiên.

## 5. Phát hiện Tọa độ và Góc (`detector.py`)
* **Core Model:** `PaddleOCR` (Cấu hình `lang="vi"`).
* **Mô hình Phân loại Góc (Angle Classification):** Sử dụng một mạng CNN nhỏ làm bộ phân loại nhị phân đa lớp (Multi-class Classifier). Đầu ra qua hàm Softmax để tính xác suất cho 4 góc quay cơ bản $\theta \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$. Dựa vào điểm tin cậy (Confidence Score), hệ thống chẩn đoán và lật lại tài liệu bị ngược $180^\circ$ do người dùng scan sai chiều.

## 6. Sắp xếp Khối văn bản (`sorter.py` & `engine.py`)
Khôi phục định dạng bảng biểu và cấu trúc 2D bằng thuật toán Phân cụm Theo ngưỡng (Threshold-based Clustering).

* Gọi tập hợp các khung chữ đã nhận diện là: $B = \{b_1, b_2, ..., b_n\}$.
* **Thuật toán chia dòng (Line Clustering):** Sắp xếp $B$ tăng dần theo tọa độ $Y_{center}$. Duyệt qua từng khung $b_i$, một khung được đưa vào dòng hiện tại $L_{current}$ nếu:
  
  $$|Y_{center}(b_i) - Y_{center}(b_{i-1})| < \theta_y$$
  
  *(với ngưỡng dung sai $\theta_y = 15$ pixels).*
* **Sắp xếp nội bộ dòng:** Các phần tử trong tập $L_{current}$ được sắp xếp lại (Sort) theo tọa độ cạnh trái $X_{min}$ tăng dần. Thuật toán này triệt tiêu hoàn toàn lỗi đứt gãy hàng ngang, tái tạo chính xác các cột phân tách nhau bằng ký tự `|`.

## 7. Hậu xử lý & Làm sạch (`post_process.py`)
* **Hàm triệt tiêu Con dấu (Stamp Noise Filter):** Con dấu hoặc hoa văn thường bị OCR nhận diện thành các ký tự đặc biệt vô nghĩa. Gọi tập ký tự lạ là $C_{special}$ và tổng ký tự là $C_{total}$. Lược bỏ chuỗi nếu tỷ lệ nhiễu vượt ngưỡng 35%:
  
  $$\frac{|C_{special}|}{|C_{total}|} > 0.35 \implies \text{Drop Text}$$
  
* **Hàm triệt tiêu Chữ ký (Signature Filter):** Đặc điểm của nét chữ ký là liên tục, loằng ngoằng và không có khoảng trắng. Lọc bỏ đoạn text nếu tồn tại từ đơn dài bất thường:
  
  $$\max_{w \in Text} (\text{length}(w)) > 8 \land w \notin \{\text{Numbers, URL, Email}\} \implies \text{Drop Text}$$
  
* **Hàm Chuẩn hóa (Rule-based Replacement):** Sử dụng Biểu thức chính quy (Regular Expressions) để bắt mẫu sai (Pattern Matching) điển hình của hệ thống đối với văn bản nhà nước (Ví dụ: `11DA` $\rightarrow$ `HÒA`, `PINT` $\rightarrow$ `PTNT`) và quy chuẩn cấu trúc ngày tháng về định dạng chuẩn `DD/MM/YYYY`.