# ==============================================================================
# TRUNG TÂM CẤU HÌNH HỆ THỐNG (SYSTEM SETTINGS)
# ==============================================================================

# ---------------------------------------------------------
# 1. ĐƯỜNG DẪN HỆ THỐNG (SYSTEM PATHS)
# ---------------------------------------------------------
MODEL_ML_PATH = "model_weights/ml_model.pkl"
MODEL_TASK_PATH = "model_weights/hand_landmarker.task"

# ---------------------------------------------------------
# 2. CẤU HÌNH LÕI THỊ GIÁC (MEDIAPIPE CORE)
# ---------------------------------------------------------
MP_NUM_HANDS = 1
MP_DETECTION_CONFIDENCE = 0.5
MP_PRESENCE_CONFIDENCE = 0.7
MP_TRACKING_CONFIDENCE = 0.5

# ---------------------------------------------------------
# 3. CẤU HÌNH KHÔNG GIAN (SPATIAL MAPPING)
# ---------------------------------------------------------
# Căn lề Active Box. Khuyến nghị lề dưới (BOTTOM) cao để tay không vướng mặt bàn
MARGIN_TOP = 20
MARGIN_BOTTOM = 180
MARGIN_LEFT = 50
MARGIN_RIGHT = 100

# ---------------------------------------------------------
# 4. CẤU HÌNH BỘ LỌC BẢO VỆ (FILTERS & KINEMATICS)
# ---------------------------------------------------------
# Lá chắn Nhiễu gai (Spike Noise)
SPIKE_MEDIAN_WINDOW = 5        # Cửa sổ trung vị (độ trễ siêu nhỏ)
SPIKE_MAX_JUMP = 150        # Vận tốc phi lý (pixel/frame) coi là nhiễu
SPIKE_PATIENCE = 5             # Sức chịu đựng số frame mất dấu liên tiếp

# Động lực học (ZUPT Kalman Filter)
KALMAN_LOCK_THRESHOLD = 40.0    # Vận tốc dưới ngưỡng này -> Chuột đứng im chống rung
KALMAN_UNLOCK_THRESHOLD = 90.0 # Lực bứt phá để chuột thoát khỏi trạng thái đứng im

# ---------------------------------------------------------
# 5. CẤU HÌNH TRẢI NGHIỆM (USER EXPERIENCE - UX)
# ---------------------------------------------------------
CONFIRM_FRAMES = 1             # Số frame liên tiếp để AI chốt 1 nhãn (Tăng lên 3 nếu hay bị chớp nhãn)
DRAG_RADIUS = 150               # Bán kính Mỏ neo chống trượt khi Click (pixel)
SCROLL_THRESHOLD = 10          # Quãng đường tay phải di chuyển để cuộn trang 1 nấc
MAX_MISSING_FRAMES = 5         # "Thời gian ân hạn" (Coasting) khi Camera mất dấu tay
SCROLL_GRACE_FRAMES = 5        # Số frame bảo lưu mỏ neo cuộn khi AI bị nhiễu
DRAG_DELAY_FRAMES = 5          # Số frame chờ chấn động cơ học
HOLD_INTENT_FRAMES = 25        # (Khoảng 0.5s) Ranh giới phân định Ý định. Chờ quá số này -> Đổ bê tông Tua x2.
