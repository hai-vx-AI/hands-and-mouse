# AI Virtual Mouse Controller 🎯

Hệ thống điều khiển chuột máy tính không chạm thông qua Camera, ứng dụng Trí tuệ Nhân tạo (Computer Vision & Machine Learning) và các bộ lọc Động lực học Không-Thời gian.

## 🧠 Kiến trúc Hệ thống (System Architecture)
- **Trích xuất Đặc trưng (Feature Extraction):** MediaPipe Hand Landmarker.
- **Phân loại Ý định (Intent Classification):** Random Forest Classifier (Dữ liệu tự huấn luyện).
- **Xử lý Tín hiệu Cơ học (Signal Processing):** 
  - Spike Noise Filter: Triệt tiêu nhiễu gai của AI.
  - ZUPT Kalman Filter: Làm mịn quỹ đạo và khóa tọa độ tuyệt đối chống rung cơ học.
  - Hysteresis & Grace Period: Cầu dao thời gian chống phân mảnh trạng thái khi nhiễu nhãn.

## ⚙️ Yêu cầu Phần cứng & Phần mềm
- Webcam (Hoạt động tốt ở 30 FPS).
- Python 3.9 -> 3.11 (Không khuyến nghị Python 3.12+ do xung đột MediaPipe).
- Hệ điều hành: Windows 10/11 (Kiến trúc tương tác OS Driver hiện tại được tối ưu cho Win32 API).

## 🚀 Hướng dẫn Cài đặt & Khởi chạy (Dành cho Người dùng/Kỹ sư)

**Bước 1: Clone mã nguồn về máy**
```bash
git clone https://github.com/hai-vx-AI/hands-and-mouse.git
cd hands-and-mouse
```

**Bước 2: Khởi tạo Môi trường Ảo (Bắt buộc để cách ly lõi hệ thống)**
```bash
python -m venv venv
```
Kích hoạt môi trường ảo trên Hệ điều hành Windows (Sử dụng Command Prompt hoặc PowerShell):
```bash
venv\Scripts\activate
```

**Bước 3: Nạp cấu hình Thư viện**
Đảm bảo đã kích hoạt môi trường ảo thành công (có chữ `(venv)` ở đầu dòng lệnh). Tiến hành nạp ma trận dependencies:
```bash
pip install -r requirements.txt
```

**Bước 4: Kích hoạt Lõi Hệ thống**
```bash
python main.py
```
*(Hệ thống sẽ tự động chiếm quyền điều khiển luồng Camera mặc định, nạp trọng số Random Forest và khởi chạy giao diện HUD).*

## 🕹️ Ma trận Cử chỉ Mặc định (Command Routing)

*   **1 Ngón tay (Trỏ):** Rê chuột tự do trên không gian 2D. Tọa độ được gọt giũa qua ZUPT Kalman Filter.
*   **2 Ngón tay (Chụm lại & Lướt):** Kéo thả (Drag & Drop). Yêu cầu quỹ đạo bứt phá khỏi vùng `DRAG_RADIUS` để mở khóa ly hợp cơ học.
*   **2 Ngón tay (Chụm lại & Giữ tĩnh):** Đổ bê tông tọa độ, giữ chuột tĩnh lặng tuyệt đối. Cơ chế này ép hệ thống từ chối mọi sự kiện xê dịch cấp thấp của Windows, thiết kế chuyên biệt để bảo vệ tính năng Tua nhanh x2 trên YouTube.
*   **3 Ngón tay:** Cuộn trang (Scroll) mượt mà dựa trên vi phân tọa độ Y. Tích hợp cơ chế Grace Period (Thời gian ân hạn) để chống đứt gãy lệnh khi AI rớt nhãn trong tíc tắc.
*   **Nắm tay (Fist):** Cầu dao An toàn (Kill Switch) - Hệ thống lập tức thả mọi mỏ neo, cắt đứt luồng tín hiệu truyền xuống Windows API và reset toàn bộ trạng thái Máy trạng thái.

## ⚠️ Điểm mù & Giới hạn Vật lý (Known Limitations)

- **Độ nhiễu Quang học:** Mạng nơ-ron lõi (MediaPipe) sẽ suy giảm nghiêm trọng độ tự tin nếu môi trường ngược sáng hoặc ánh sáng yếu, dẫn đến mất nhãn đầu vào cho Random Forest.
- **Trễ Cơ sinh học (Biomechanical Jerk):** Thao tác Click/Drag được thiết kế với một độ trễ vật lý có chủ đích (~0.15s thông qua `DRAG_DELAY_FRAMES`). Đây không phải là nút thắt cổ chai của thuật toán, mà là khoảng thời gian hệ thống tự làm "mù" chính nó để triệt tiêu chấn động gân tay và cổ tay khi con người thực hiện thao tác bóp cò trên không trung. Mọi sự xê dịch trong khung thời gian này đều bị tước bỏ.  
