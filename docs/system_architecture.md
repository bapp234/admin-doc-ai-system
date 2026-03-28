# ĐẶC TẢ KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE SPECIFICATION)

## 1. Module OCR (src/modules/ocr)
**Mục tiêu:** Xử lý hình ảnh tài liệu thô, nắn chỉnh tự động và chuyển đổi thành văn bản có cấu trúc 2D.

### 1.1. Lớp `OCREngine` (engine.py)
* **Chức năng:** Điều phối toàn bộ luồng pipeline OCR.
* **Khởi tạo (`__init__`):** * Nạp `TextRecognizer` (VietOCR) và `LayoutAnalyzer` (docTR).
  * `use_gpu` (bool): Kích hoạt chạy trên card đồ họa rời nếu có.
* **Phương thức `extract(img_array: np.ndarray) -> dict`:**
  * **Input:** Ma trận điểm ảnh NumPy (đọc bằng OpenCV).
  * **Xử lý:**
    1. Gọi phân tích bố cục để lấy danh sách bounding boxes.
    2. Thuật toán xoay ảnh tự động (Auto-rotation) 90/270 độ nếu phát hiện ảnh bị lật dọc (ngưỡng phát hiện < 3 boxes).
    3. Bộ lọc nhiễu hình học: Loại bỏ các box có kích thước nhỏ hơn 10px hoặc có tỷ lệ dị dạng (chiều cao > 3 lần chiều rộng).
    4. Tái tạo cấu trúc 2D (2D Layout Reconstruction): Gom nhóm các text box nằm trên cùng trục Y (sai số 15px) và sắp xếp từ trái sang phải dựa trên tọa độ X nhỏ nhất (`left_x`).
  * **Output:** Dictionary chứa:
    * `text` (str): Văn bản hoàn chỉnh, các cột phân cách bằng dấu `|`.
    * `confidence` (float): Tỷ lệ tự tin trung bình của bộ nhận dạng (%).
    * `rotated_angle` (int): Góc xoay đã áp dụng để nắn ảnh.