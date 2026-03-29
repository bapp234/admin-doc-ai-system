# ĐẶC TẢ KIẾN TRÚC VÀ THUẬT TOÁN - MODULE TIỀN XỬ LÝ 
**Dự án:** Admin Doc AI System  
**Phân hệ:** Tiền xử lý Hình ảnh (Preprocessing Pipeline)

---

## 1. Tổng quan Module Tiền xử lý
Module Tiền xử lý đóng vai trò quyết định trong việc đảm bảo chất lượng đầu vào cho các mô hình AI phía sau. Hệ thống sử dụng một chuỗi (Pipeline) các phép biến đổi thị giác máy tính (Computer Vision) thuần túy để căn chỉnh, làm sạch nhiễu, cân bằng sáng và bóc tách các thành phần không mong muốn (như mộc đỏ) mà không cần tiêu tốn tài nguyên chạy Deep Learning.

## 2. Phát hiện và Căn chỉnh Tự động (`alignment.py` & `detector.py`)
Đảm nhiệm việc phát hiện góc chụp và tự động cắt cúp khung giấy (Auto-scan) đối với các hình ảnh được chụp bằng thiết bị di động.

* **Thuật toán Phân loại Nền (`detector.py`):** Sử dụng phép toán hình thái học (Morphological Dilation) để xóa bỏ các chi tiết chữ, giữ lại độ chói của nền giấy. Tính toán độ lệch chuẩn (Standard Deviation) $\sigma$ của ma trận cường độ sáng. Nếu $\sigma < 15$, hệ thống xác định đây là ảnh Scan tiêu chuẩn; ngược lại là ảnh Camera.
* **Thuật toán Auto-Scan (`alignment.py`):**
  1. **Lọc nhiễu bảo toàn cạnh (Bilateral Filter):** Giảm nhiễu không gian nhưng vẫn giữ được độ sắc nét của viền giấy.
  2. **Phát hiện biên (Canny Edge Detection):** Sử dụng ngưỡng đạo hàm gradient $(30, 200)$ để tạo ma trận nhị phân đường viền.
  3. **Xấp xỉ Đa giác (Polygonal Approximation):** Trích xuất các Contours và dùng thuật toán Ramer-Douglas-Peucker để tìm tứ giác lớn nhất có diện tích thỏa mãn $S_{contour} > 0.2 \times S_{image}$.
  4. **Biến đổi Phối cảnh (Perspective Transform):** Áp dụng công thức ma trận tương tự Module OCR để nắn phẳng tờ giấy bị chụp nghiêng.

## 3. Hiệu chỉnh Góc Xoay Trục (`deskew.py`)
Xử lý các tài liệu scan bị lệch góc nhỏ (dưới 10 độ).

* **Thuật toán Phân tích Hình chiếu (Projection Profile Analysis):**
  Thay vì dùng Hough Transform tốn kém, hệ thống xoay thử nghiệm ảnh góc $\theta \in [-10^\circ, 10^\circ]$ với bước nhảy $0.5^\circ$. Tại mỗi góc, tính phương sai (Variance) của tổng cường độ pixel theo các hàng ngang:
  $$\theta_{best} = \arg\max_{\theta} \left( \text{Var}\left(\sum_{x} I_{\theta}(x, y)\right) \right)$$
  *(Nguyên lý: Khi văn bản nằm ngang tuyệt đối, sự chênh lệch (phương sai) giữa dòng có chữ (đen) và dòng khoảng trắng (trắng) sẽ đạt mức cực đại).*

## 4. Lọc Nhiễu và Khử Bóng râm (`filters.py`)
* **Khử nhiễu không gian (Denoising):** Sử dụng thuật toán Non-Local Means (NLM) `cv2.fastNlMeansDenoising`. Khác với Gaussian Blur làm mờ ảnh, NLM tìm kiếm và tính trung bình các mảng pixel có cấu trúc kết cấu (texture) giống nhau trên toàn bộ bức ảnh.
* **Xử lý bóng đổ (Shadow Removal):**
  1. Phân rã nền bằng phép Dilate và Median Blur kích thước cực đại (kernel 21x21).
  2. Lấy cường độ ảnh gốc trừ đi nền để chuẩn hóa (Background Subtraction).
  3. Sử dụng phép biến đổi Gamma (Gamma Correction) với $\gamma = 0.5$ đối với các ảnh bị lóa sáng quá mức (Mean Intensity > 230) để phục hồi mật độ mực in:
     $$V_{out} = 255 \times \left( \frac{V_{in}}{255} \right)^{\frac{1}{\gamma}}$$

## 5. Bóc tách và Khử Mộc đỏ (`stamp.py`)
Mộc đỏ của cơ quan nhà nước thường đè lên chữ, gây nhiễu nghiêm trọng cho thuật toán binarization và AI nhận diện.

* **Thuật toán Không gian màu HSV:** Chuyển ma trận ảnh từ BGR sang HSV (Hue, Saturation, Value) để tách biệt hoàn toàn thông tin màu sắc khỏi cường độ sáng.
* **Mặt nạ Ngưỡng Kép (Dual-Threshold Masking):** Do màu đỏ nằm ở hai đầu của phổ Hue, hệ thống sử dụng phép hội (Bitwise OR) của hai dải phổ: $H \in [0, 10] \cup [170, 180]$.
* **Tinh chỉnh Hình thái học (Morphology):** Dùng phép Mở (Morphology Open) với ma trận Ellipse $5 \times 5$ để loại bỏ nhiễu hạt, kết hợp thuật toán Connected Components để làm sạch hoàn toàn vùng chữ bị mộc che khuất, trả lại nền trắng cho OCR.