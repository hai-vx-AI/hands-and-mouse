'''
# ==============================================================================
# KIẾN TRÚC HỆ THỐNG: TẦNG HẬU XỬ LÝ & ĐIỀU KHIỂN THỰC THỂ (POST-PROCESSING & CONTROL LAYER)
# ==============================================================================
# BẢN CHẤT CỐT LÕI:
# Phiên dịch phán đoán thô (Raw Predictions) từ AI và tọa độ rung lắc từ Camera 
# thành các luồng lệnh thao tác chuột (Mouse Actions) mượt mà, chính xác và an toàn trên HĐH Windows.
#
# CẤU TRÚC ĐƯỜNG ỐNG (PIPELINE) GỒM 5 MODULE ĐỘC LẬP:
#
# [MODULE 1] SPATIAL MAPPING (Đồng bộ Không gian Tọa độ)
# - Xây dựng "Vùng hoạt động" (Active Box) trên Camera để loại bỏ vùng rìa biên (Deadzones).
# - Áp dụng thuật toán nội suy (Interpolation) để biến đổi hệ quy chiếu 4:3 của Webcam 
#   sang ma trận 16:9 của màn hình máy tính (ví dụ: 1920x1080) mà không bị méo tỷ lệ.
#
# [MODULE 2] SIGNAL SMOOTHING & KINEMATICS (Lọc nhiễu & Động học Tọa độ)
# - Triển khai Kalman Filter (hoặc Exponential Moving Average - EMA).
# - Hấp thụ các xung nhiễu (Jitter) tọa độ đầu ngón tay. Đảm bảo khi tay người đứng im, 
#   chuột trên màn hình phải ghim chặt tuyệt đối, không rung lắc.
#
# [MODULE 3] TEMPORAL STATE MACHINE (Máy trạng thái Thời gian & Chống dội)
# - Thiết lập cơ chế Debouncing cho các luồng tín hiệu rời rạc (Clicks/Scrolls).
# - Ép ràng buộc: Phán đoán từ AI phải duy trì ổn định trong N frames liên tiếp (VD: 3 frames) 
#   thì mới được xác nhận là một lệnh hợp lệ (Triệt tiêu False Positives 1-frame).
# - Thiết lập Cooldown/Hysteresis Lock (Khóa trạng thái) sau mỗi cú Click để chống Spam.
#
# [MODULE 4] OS EXECUTION (Thực thi Lệnh Hệ điều hành)
# - Bọc thư viện PyAutoGUI vào các hàm điều khiển an toàn.
# - Thiết lập Cầu dao khẩn cấp (Kill Switch/Fail-safe): Tự động ngắt mọi thao tác 
#   khi nhận nhãn 'Reset' hoặc khi tay người rời khỏi tầm nhìn Camera.
#
# [MODULE 5] THE ORCHESTRATOR (Vòng lặp Thời gian thực - Main Loop)
# - Điểm hội tụ cuối cùng. Kéo luồng Camera -> MediaPipe Tasks API -> Đặc trưng hình học ->
#   Random Forest Inference (.pkl) -> Bơm vào Module 1, 2, 3 -> Thực thi tại Module 4.
# ==============================================================================
'''

import ctypes
import math
import time
import pyautogui
import numpy as np
import cv2
from collections import deque


class SpikeNoiseFilter:
    def __init__(self, median_window=3, max_jump_pixels=150, max_patience_frames=5):
        """
        LÁ CHẮN TIỀN XỬ LÝ NHIỄU GAI (PRE-PROCESSING ARMOR)
        
        Args:
            median_window: Số phần tử của mảng Trung vị. 
                           N=3 hoặc 5 là tối ưu. Lớn hơn sẽ gây "độ trễ kéo lê" (Lag).
            max_jump_pixels: Vận tốc phi lý (pixel/frame). Nếu tọa độ nhảy vọt qua số này, coi là Nhiễu.
            max_patience_frames: Sức chịu đựng. Nếu quá N frame liên tiếp đều là "Nhiễu", 
                                 hệ thống sẽ nhượng bộ và coi đó là hiện thực mới.
        """
        self.window_size = median_window
        self.max_jump = max_jump_pixels
        self.max_patience = max_patience_frames
        
        # Bộ nhớ Trung vị
        self.buffer_x = deque(maxlen=median_window)
        self.buffer_y = deque(maxlen=median_window)
        
        # Bộ nhớ Động năng
        self.last_valid_x = None
        self.last_valid_y = None
        self.strike_counter = 0

    def filter(self, raw_screen_x, raw_screen_y):
        if self.last_valid_x is None:
            self.last_valid_x = raw_screen_x
            self.last_valid_y = raw_screen_y
            self._append_to_buffer(raw_screen_x, raw_screen_y)
            return raw_screen_x, raw_screen_y

        jump_dist = math.hypot(raw_screen_x - self.last_valid_x, raw_screen_y - self.last_valid_y)
        
        if jump_dist > self.max_jump:
            self.strike_counter += 1
            
            if self.strike_counter <= self.max_patience:
                safe_x, safe_y = self.last_valid_x, self.last_valid_y
            else:
                safe_x, safe_y = raw_screen_x, raw_screen_y
                self.last_valid_x = raw_screen_x
                self.last_valid_y = raw_screen_y
                self.strike_counter = 0 
        else:
            safe_x, safe_y = raw_screen_x, raw_screen_y
            self.last_valid_x = raw_screen_x
            self.last_valid_y = raw_screen_y
            self.strike_counter = 0

        self._append_to_buffer(safe_x, safe_y)
        
        median_x = int(np.median(self.buffer_x))
        median_y = int(np.median(self.buffer_y))

        return median_x, median_y

    def _append_to_buffer(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)

    def reset(self):
        """Gọi hàm này khi tay cất khỏi Camera để xóa toàn bộ bộ nhớ"""
        self.buffer_x.clear()
        self.buffer_y.clear()
        self.last_valid_x = None
        self.last_valid_y = None
        self.strike_counter = 0

class SpatialMapper:
    def __init__(self, cam_w=640, cam_h=480, margin_top=20, margin_bottom=150, margin_left=50, margin_right=100):
        """
        Module 1: Đồng bộ Không gian Tọa độ (Kiến trúc Phi đối xứng)
        
        Args:
            cam_w, cam_h: Kích thước khung hình camera.
            margin_top: Khoảng cách từ viền trên camera đẩy xuống (pixel).
            margin_bottom: Khoảng cách từ viền dưới camera đẩy lên (pixel).
            margin_left: Khoảng cách từ viền trái camera đẩy vào (pixel).
            margin_right: Khoảng cách từ viền phải camera đẩy vào (pixel).
        """
        self.cam_w = cam_w
        self.cam_h = cam_h
        
        self.m_top = margin_top
        self.m_bottom = margin_bottom
        self.m_left = margin_left
        self.m_right = margin_right
        
        self.screen_w, self.screen_h = pyautogui.size()
        
        print(f"[MODULE 1] Spatial Mapper Initialized (Asymmetrical Box).")
        print(f" - Screen Space: {self.screen_w}x{self.screen_h}")
        print(f" - Active Box [T: {self.m_top}, B: {self.m_bottom}, L: {self.m_left}, R: {self.m_right}]")

    def map_coordinates(self, hand_x_pixel, hand_y_pixel):
        """
        Nội suy tuyến tính dải tọa độ giới hạn trên Camera sang dải phân giải Màn hình.
        """
        screen_x = np.interp(
            hand_x_pixel, 
            (self.m_left, self.cam_w - self.m_right), 
            (0, self.screen_w)
        )
        
        screen_y = np.interp(
            hand_y_pixel, 
            (self.m_top, self.cam_h - self.m_bottom), 
            (0, self.screen_h)
        )
        
        return screen_x, screen_y

    def get_box_coordinates(self):
        """Hàm bổ trợ để xuất tọa độ vẽ HUD bên main.py"""
        pt1 = (self.m_left, self.m_top)
        pt2 = (self.cam_w - self.m_right, self.cam_h - self.m_bottom)
        return pt1, pt2

# ==============================================================================
# [MODULE 2] ĐỘNG HỌC TỰ TRỊ (AUTONOMOUS KINEMATICS & ZUPT)
# ==============================================================================


class ZUPT_KalmanFilter:
    def __init__(self, lock_threshold=15.0, unlock_threshold=40.0):
        """
        [MODULE 2] LỌC ĐỘNG HỌC & HYSTERESIS ZUPT
        """
        self.kalman = cv2.KalmanFilter(4, 2)
        
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1],
            [0, 0, 1, 0], [0, 0, 0, 1]
        ], np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0]
        ], np.float32)

        self.loose_noise = np.eye(4, dtype=np.float32) * 0.05
        self.strict_noise = np.eye(4, dtype=np.float32) * 0.0015
        
        self.kalman.processNoiseCov = self.loose_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        
        self.is_initialized = False
        
        self.is_locked = False 
        
        self.lock_threshold = lock_threshold    
        self.unlock_threshold = unlock_threshold 
        
        self.prev_cam_x = 0
        self.prev_cam_y = 0

    def update(self, cam_x, cam_y):
        measurement = np.array([[np.float32(cam_x)], [np.float32(cam_y)]])

        if not self.is_initialized:
            self.kalman.statePre = np.array([[measurement[0,0]], [measurement[1,0]], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[measurement[0,0]], [measurement[1,0]], [0], [0]], np.float32)
            self.is_initialized = True
            
            self.prev_cam_x = cam_x
            self.prev_cam_y = cam_y
            return int(cam_x), int(cam_y), False

        raw_v_x = cam_x - self.prev_cam_x
        raw_v_y = cam_y - self.prev_cam_y
        raw_velocity = math.hypot(raw_v_x, raw_v_y)
        
        self.prev_cam_x = cam_x
        self.prev_cam_y = cam_y

        if self.is_locked:
            if raw_velocity > self.unlock_threshold:
                self.is_locked = False
        else:
            if raw_velocity < self.lock_threshold:
                self.is_locked = True

        if self.is_locked:
            self.kalman.processNoiseCov = self.strict_noise
        else:
            self.kalman.processNoiseCov = self.loose_noise

        self.kalman.predict()
        estimated_state = self.kalman.correct(measurement)
        
        smooth_x = int(estimated_state[0, 0])
        smooth_y = int(estimated_state[1, 0])
        
        return smooth_x, smooth_y, self.is_locked
        
# ==============================================================================
# [MODULE 3] ĐỘNG CƠ CLICK BẤT BIẾN TỶ LỆ (SCALE-INVARIANT CLICK ENGINE)
# ==============================================================================

class GestureClickEngine:
    def __init__(self):
        """
        [MODULE 3] ĐỘNG CƠ CLICK & DRAG THUẦN CƠ HỌC
        Tuyệt đối không dùng thời gian (Cooldown/Hold time).
        """
        self.is_mouse_down = False 

    def process_click(self, current_label, click_labels, move_labels):
        """
        click_labels: Mảng chứa các nhãn 2 ngón (Kích hoạt đè chuột)
        move_labels: Mảng chứa các nhãn 1 ngón (Kích hoạt nhả chuột)
        """
        if current_label in click_labels:
            if not self.is_mouse_down:
                pyautogui.mouseDown()
                self.is_mouse_down = True
                return "MOUSE_DOWN_STARTED"
            else:
                return "DRAGGING"

        elif current_label in move_labels:
            if self.is_mouse_down:
                pyautogui.mouseUp()
                self.is_mouse_down = False
                return "MOUSE_UP_RELEASED"
            
        return "IDLE"
    

# ==============================================================================
# [MODULE 3.5] TEMPORAL LABEL CONFIRMER (N-frame Debouncing)
# ==============================================================================

class LabelConfirmer:
    def __init__(self, confirm_frames=3):
        """
        Chỉ xác nhận nhãn khi nó xuất hiện liên tục đủ N frames.
        
        Args:
            confirm_frames (int): Số frames liên tiếp cần thiết để xác nhận.
                                  Tăng số này → phản ứng chậm hơn nhưng ổn định hơn.
                                  Giảm số này → phản ứng nhanh hơn nhưng dễ false positive.
        """
        self.confirm_frames = confirm_frames
        self.buffer = deque(maxlen=confirm_frames)  
        self.confirmed_label = "None"

    def update(self, raw_label):
        """
        Nhận nhãn thô từ AI, trả về nhãn đã được xác nhận.
        
        Returns:
            str: Nhãn xác nhận nếu đủ N frames liên tiếp, 
                 ngược lại giữ nguyên nhãn xác nhận trước đó.
        """
        self.buffer.append(raw_label)

        if len(self.buffer) == self.confirm_frames:
            if len(set(self.buffer)) == 1: 
                self.confirmed_label = raw_label

        return self.confirmed_label

    def reset(self):
        """Xóa buffer khi tay rời camera."""
        self.buffer.clear()
        self.confirmed_label = "None"
        
class OSDriver:
    def __init__(self):
        """
        [MODULE 4] OS EXECUTION & FAIL-SAFE CONTROLLER
        Người gác cổng cuối cùng trước khi lệnh được nạp vào Hệ điều hành Windows.
        """
        pyautogui.PAUSE = 0
        
        pyautogui.FAILSAFE = False
        
        self.is_killed = False
        
        print("[MODULE 4] OS Driver Initialized. Safety protocols engaged.")

    def evaluate_kill_switch(self, has_hand, current_ai_label):
        """
        Kiểm duyệt An ninh: Chạy mỗi frame trước khi cho phép bất kỳ hành động nào.
        has_hand (bool): True nếu MediaPipe tìm thấy tay.
        current_ai_label (str): Nhãn từ mô hình Random Forest.
        """
        if not has_hand or current_ai_label == "Reset":
            if not self.is_killed:
                print("\n[SYSTEM ALERT] CẦU DAO ĐÃ ĐÓNG! Hệ thống đang bị khóa.")
                self.is_killed = True
                pyautogui.mouseUp() 
                
            return True
            
        else:
            if self.is_killed:
                print("[SYSTEM ALERT] CẦU DAO ĐÃ MỞ. Khôi phục quyền điều khiển.\n")
                self.is_killed = False
                
            return False

    # ==========================================
    # CÁC HÀM THỰC THI (CHỈ CHẠY KHI CẦU DAO MỞ)
    # ==========================================

    def execute_move(self, x, y):
        if self.is_killed: return
        print(f">>> [OS PROBE] ĐANG BƠM LỆNH XUỐNG WINDOWS: X={x} | Y={y}")
        pyautogui.moveTo(int(x), int(y), _pause=False)

    def execute_click(self):
        """Thực thi thao tác Nhấp chuột"""
        if self.is_killed: return
        pyautogui.click()

    def execute_scroll(self, delta_y):
        """
        Thực thi Cuộn trang Tương đối (Relative Scroll).
        delta_y âm -> Tay kéo lên -> Cuộn lên.
        delta_y dương -> Tay kéo xuống -> Cuộn xuống.
        """
        if self.is_killed: return
        
        scroll_amount = int(delta_y * -5.0) 
        pyautogui.scroll(scroll_amount)
    def execute_hotkey(self, *keys):
        pyautogui.hotkey(*keys)

