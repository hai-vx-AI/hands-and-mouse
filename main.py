import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
import time
import math

import config  
from src.post_processing import SpatialMapper, ZUPT_KalmanFilter, GestureClickEngine, OSDriver, LabelConfirmer, SpikeNoiseFilter


def run_system():
    print("[SYSTEM] Đang nạp Bộ não Trí tuệ Nhân tạo...")
    try:
        rf_model = joblib.load(config.MODEL_ML_PATH)
        print(f"[SUCCESS] Đã nạp thành công: {config.MODEL_ML_PATH}")
    except Exception as e:
        raise Exception(f"[FATAL ERROR] Không tìm thấy file Model: {e}")

    print("[SYSTEM] Đang nạp Lõi thị giác MediaPipe Tasks API...")
    base_options = python.BaseOptions(model_asset_path=config.MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=config.MP_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MP_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MP_TRACKING_CONFIDENCE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("[FATAL ERROR] Không thể chiếm quyền Camera.")

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract_features(hand_landmarks_list):
        features = []
        wrist_x = hand_landmarks_list[0].x
        wrist_y = hand_landmarks_list[0].y
        for i in range(1, 21):
            features.append(hand_landmarks_list[i].x - wrist_x)
            features.append(hand_landmarks_list[i].y - wrist_y)
        return np.array(features).reshape(1, -1)

    mapper = SpatialMapper(
        cam_w=raw_w, cam_h=raw_h, 
        margin_top=config.MARGIN_TOP, 
        margin_bottom=config.MARGIN_BOTTOM, 
        margin_left=config.MARGIN_LEFT, 
        margin_right=config.MARGIN_RIGHT
    )
    spike_filter = SpikeNoiseFilter(
        median_window=config.SPIKE_MEDIAN_WINDOW, 
        max_jump_pixels=config.SPIKE_MAX_JUMP, 
        max_patience_frames=config.SPIKE_PATIENCE
    )
    tracker = ZUPT_KalmanFilter(
        lock_threshold=config.KALMAN_LOCK_THRESHOLD, 
        unlock_threshold=config.KALMAN_UNLOCK_THRESHOLD
    )

    clicker = GestureClickEngine()
    os_driver = OSDriver()
    confirmer = LabelConfirmer(confirm_frames=config.CONFIRM_FRAMES)

    CMD_MOVE = ['Move']           
    CMD_CLICK = ['Scroll_2']      
    CMD_SCROLL = ['Scroll_3']     
    CMD_RESET = ['Reset']        

    pTime = 0
    drag_anchor = None
    is_drag_unlocked = False      
    drag_frames_count = 0 
    scroll_grace_count = 0    
    is_hold_mode = False
    scroll_anchor_y = None 
    is_locked = False
    missing_frames_count = 0
    prev_sent_x, prev_sent_y = -1, -1

    print("\n[SYSTEM] HỆ THỐNG ĐÃ LÊN ĐÈN. ĐANG BẮT ĐẦU LUỒNG THỰC THI CHÍNH...")

    while True:
        success, img = cap.read()
        if not success: continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)
        
        current_label = "None"
        click_status = "IDLE"

        # ==========================================
        # NHÁNH 1: KHI TAY XUẤT HIỆN
        # ==========================================
        if detection_result.hand_landmarks:
            missing_frames_count = 0
            hand_lms = detection_result.hand_landmarks[0]
            features = extract_features(hand_lms)
            
            predicted_label = str(rf_model.predict(features)[0]).strip()
            confirmed_label = confirmer.update(predicted_label)
            current_label = f"{predicted_label} → {confirmed_label}"

            is_killed = os_driver.evaluate_kill_switch(
                has_hand=True,
                current_ai_label=("Reset" if predicted_label in CMD_RESET else "Safe")
            )

            if not is_killed:
                idx_x_pixel = int(hand_lms[8].x * raw_w)
                idx_y_pixel = int(hand_lms[8].y * raw_h)
                
                screen_x, screen_y = mapper.map_coordinates(idx_x_pixel, idx_y_pixel)
                clean_x, clean_y = spike_filter.filter(screen_x, screen_y)
                smooth_x, smooth_y, is_locked = tracker.update(screen_x, screen_y)

                # ==========================================
                # KIẾN TRÚC PHÂN LUỒNG MỆNH LỆNH (HYBRID ROUTING)
                # ==========================================
                
                def safe_execute_move(x, y):
                    nonlocal prev_sent_x, prev_sent_y
                    ix, iy = int(x), int(y)
                    if ix != prev_sent_x or iy != prev_sent_y:
                        os_driver.execute_move(ix, iy)
                        prev_sent_x, prev_sent_y = ix, iy

                if (confirmed_label in CMD_MOVE) or (confirmed_label in CMD_CLICK):                
                    click_status = clicker.process_click(
                        current_label=confirmed_label, 
                        click_labels=CMD_CLICK, 
                        move_labels=CMD_MOVE
                    )

                    if click_status == "MOUSE_DOWN_STARTED":
                        drag_anchor = (smooth_x, smooth_y)
                        is_drag_unlocked = False
                        
                    elif click_status == "DRAGGING":
                        if drag_anchor is not None:

                            if not is_drag_unlocked:
                                drag_dist = math.hypot(smooth_x - drag_anchor[0], smooth_y - drag_anchor[1])

                                if drag_dist > config.DRAG_RADIUS:
                                    is_drag_unlocked = True
                                    drag_frames_count = 0  
                                    is_hold_mode = False

                                else:
                                    drag_frames_count += 1

                                    if drag_frames_count == config.HOLD_INTENT_FRAMES and not is_hold_mode:
                                        is_hold_mode = True
                                        os_driver.execute_hotkey('shift', '.') 

                            if is_drag_unlocked:
                                safe_execute_move(smooth_x, smooth_y)

                    elif click_status == "MOUSE_UP_RELEASED":
                        if is_hold_mode:
                            os_driver.execute_hotkey('shift', ',') 
                        
                        drag_anchor = None
                        is_drag_unlocked = False
                        drag_frames_count = 0
                        is_hold_mode = False

                    elif click_status == "IDLE":
                        safe_execute_move(smooth_x, smooth_y)

                if confirmed_label in CMD_SCROLL:
                    scroll_grace_count = 0 
                    
                    if scroll_anchor_y is None:
                        scroll_anchor_y = clean_y 
                    else:
                        delta_y = clean_y - scroll_anchor_y
                        if abs(delta_y) > config.SCROLL_THRESHOLD:
                            os_driver.execute_scroll(delta_y)
                            scroll_anchor_y = clean_y            
                else:

                    if scroll_anchor_y is not None:
                        scroll_grace_count += 1
                        if scroll_grace_count > config.SCROLL_GRACE_FRAMES:
                            scroll_anchor_y = None
                            scroll_grace_count = 0
            
            HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),         
                (0, 5), (5, 6), (6, 7), (7, 8),         
                (5, 9), (9, 10), (10, 11), (11, 12),    
                (9, 13), (13, 14), (14, 15), (15, 16),  
                (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
            ]
            pixel_landmarks = []
            for lm in hand_lms:
                px, py = int(lm.x * raw_w), int(lm.y * raw_h)
                pixel_landmarks.append((px, py))
                cv2.circle(img, (px, py), 5, (0, 255, 255), -1)

            for connection in HAND_CONNECTIONS:
                start_point = pixel_landmarks[connection[0]]
                end_point = pixel_landmarks[connection[1]]
                cv2.line(img, start_point, end_point, (255, 0, 255), 2)

        # ==========================================
        # NHÁNH 2: KHI MẤT DẤU TAY HOÀN TOÀN
        # ==========================================
        else:
            missing_frames_count += 1
            if missing_frames_count >= config.MAX_MISSING_FRAMES:
                os_driver.evaluate_kill_switch(has_hand=False, current_ai_label="None")
                confirmer.reset() 
                tracker.is_initialized = False 
                spike_filter.reset()
                click_status = "NO_HAND"
                scroll_anchor_y = None
                drag_anchor = None

        # ==========================================
        # MODULE RENDER VÀ TELEMETRY TỔNG
        # ==========================================
        cv2.rectangle(img, 
                    (mapper.m_left, mapper.m_top), 
                    (raw_w - mapper.m_right, raw_h - mapper.m_bottom), 
                    (255, 0, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f'AI Label: {current_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f'Click Engine: {click_status}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f'ZUPT Locked: {is_locked}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f'Hold Mode: {is_hold_mode} ({drag_frames_count}f)', 
            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Kien truc Chuot ao - He thong Trung tam", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()