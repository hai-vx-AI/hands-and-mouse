import os
import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CẤU HÌNH BIẾN TOÀN CỤC
# ==========================================
MODEL_PATH = 'hand_landmarker.task'
CSV_OUTPUT_PATH = 'data_output.csv'
DATASET_ROOT = r'D:\.vscode\hand_and_mouse\hagrid-sample-30k-384p\hagrid_30k' # Trỏ đến thư mục chứa các folder train_val_...

LIMIT_PER_CLASS = 1500

TARGET_LABELS = {
    'Reset': ['train_val_palm', 'train_val_stop', 'train_val_stop_inverted'],
    'Move': ['train_val_one'],
    'Scroll_2': ['train_val_two_up', 'train_val_two_up_inverted', 'train_val_peace', 'train_val_peace_inverted'],
    'Scroll_3': ['train_val_three', 'train_val_three2'],
    'Noise': ['train_val_call', 'train_val_dislike', 'train_val_fist', 'train_val_four', 'train_val_like', 'train_val_mute', 'train_val_ok', 'train_val_rock']
}

# ==========================================
# MODULE KHỞI TẠO (PHASE 1)
# ==========================================
def create_hand_landmarker(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)

def preprocess_data_phase1():
    f = open(CSV_OUTPUT_PATH, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    header = ['Label']
    for i in range(1, 21): header.extend([f'x{i}', f'y{i}'])
    writer.writerow(header)
    return f, writer

# ==========================================
# MODULE LÕI: QUÉT DỮ LIỆU & TRÍCH XUẤT (PHASE 2 -> 5)
# ==========================================
def extract_and_process_data(landmarker, csv_writer):
    class_counters = {key: 0 for key in TARGET_LABELS.keys()}
    print("\n[INFO] Khởi động động cơ ETL. Bắt đầu quét dữ liệu...")
    
    # PHASE 2 & 3: Quét cây thư mục và Lọc nhãn
    for folder_name in os.listdir(DATASET_ROOT):
        folder_path = os.path.join(DATASET_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        system_label = None
        for sys_lbl, raw_lbls in TARGET_LABELS.items():
            if folder_name in raw_lbls:
                system_label = sys_lbl
                break
                
        if system_label is None:
            continue
            
        print(f"[PROCESS] Đang xử lý: {folder_name} -> Nhãn đích: {system_label}")
        
        for file_name in os.listdir(folder_path):
            if class_counters[system_label] >= LIMIT_PER_CLASS:
                break # Đã gom đủ số lượng cho class này
                
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(folder_path, file_name)
            
            # PHASE 4: Đọc ảnh và Suy luận AI
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)
            
            # PHASE 5: Chuẩn hóa không gian và Ghi đĩa
            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]            
                wrist_x = hand_landmarks[0].x
                wrist_y = hand_landmarks[0].y
                
                row_data = [system_label]    
                for i in range(1, 21):
                    rel_x = hand_landmarks[i].x - wrist_x
                    rel_y = hand_landmarks[i].y - wrist_y
                    # Chặt cụt ở 6 chữ số thập phân để tối ưu dung lượng file
                    row_data.extend([round(rel_x, 6), round(rel_y, 6)])
                
                csv_writer.writerow(row_data)
                class_counters[system_label] += 1

    print("\n[SUCCESS] BÁO CÁO CẤU TRÚC DATASET HOÀN THIỆN:")
    for sys_lbl, count in class_counters.items():
        print(f" - {sys_lbl}: {count} dòng dữ liệu")

# ==========================================
# THỰC THI LUỒNG CHÍNH
# ==========================================
if __name__ == "__main__":
    print("[SYSTEM] Đang nạp mô hình MediaPipe Tasks API vào RAM...")
    ai_model = create_hand_landmarker(MODEL_PATH)
    
    print("[SYSTEM] Đang khởi tạo tệp tin đích...")
    csv_file, csv_writer = preprocess_data_phase1()
    
    try:
        extract_and_process_data(ai_model, csv_writer)
    except KeyboardInterrupt:
        print("\n[WARNING] Quá trình bị ngắt bởi người dùng.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Hệ thống sụp đổ: {e}")
    finally:
        csv_file.close()
        ai_model.close()
        print("[SYSTEM] Đã ngắt kết nối an toàn. Kiểm tra file CSV đầu ra.")