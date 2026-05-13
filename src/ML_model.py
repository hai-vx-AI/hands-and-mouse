'''
# ==============================================================================
# KIẾN TRÚC HỆ THỐNG: QUY TRÌNH HUẤN LUYỆN MÔ HÌNH (ML TRAINING PIPELINE)
# ==============================================================================
# BẢN CHẤT CỐT LÕI: 
# Chuyển đổi Không gian đặc trưng hình học (40 chiều tọa độ tương đối) 
# thành các Ranh giới quyết định logic (Decision Boundaries) để nhận diện cử chỉ.
#
# QUY TRÌNH THỰC THI (TOP-DOWN) GỒM 5 GIAI ĐOẠN:
#
# [PHASE 1] DATA INGESTION & SPLITTING (Nạp và Phân mảnh Dữ liệu)
# - Nạp bộ dữ liệu 'hand_gestures_normalized.csv' từ ổ cứng lên RAM (Pandas).
# - Bóc tách Ma trận Đặc trưng X (40 cột tọa độ) và Vector Nhãn đích y (1 cột Label).
# - Cắt trích xuất dữ liệu thành 2 phần độc lập: Train Set (để học) và Test Set (để thi).
#   *Ràng buộc: Sử dụng cơ chế Stratified-Split để bảo toàn tỷ lệ phân phối ban đầu của các nhãn.
#
# [PHASE 2] ALGORITHM INITIALIZATION (Khởi tạo Cỗ máy Toán học)
# - Khởi tạo mô hình Random Forest Classifier.
# - Thiết lập Siêu tham số (Hyperparameters): Số lượng cây (n_estimators), độ sâu (max_depth).
# - Thiết lập "Tự động cân bằng" (class_weight='balanced') làm hàng rào phòng ngự đầu tiên 
#   để chống lại sự chênh lệch số lượng của lớp Noise so với các lớp khác.
#
# [PHASE 3] MODEL FITTING (Đóng gói Tri thức)
# - Bơm dữ liệu Train (X_train, y_train) vào mạng lưới thuật toán.
# - Hệ thống tự động phân nhánh, thiết lập hàng triệu luật logic IF-ELSE dựa trên Entrophy.
#
# [PHASE 4] VALIDATION & DIAGNOSTICS (Kiểm định và Khám nghiệm)
# - Chạy suy luận mù (Blind Inference) trên tập Test (X_test).
# - Xuất báo cáo Tổng cục: Độ chính xác tuyệt đối (Accuracy).
# - Xuất khám nghiệm Cục bộ: Classification Report (Precision, Recall, F1-Score).
#   *Mục tiêu tối thượng: Soi chiếu ma trận nhầm lẫn (Confusion Matrix) để xem các lớp cốt lõi 
#   như Move, Scroll có bị lớp Noise "nuốt chửng" hay không.
#
# [PHASE 5] SERIALIZATION & EXPORT (Đóng băng và Xuất xưởng)
# - Đóng gói toàn bộ cấu trúc cây quyết định thành file nhị phân (.pkl).
# - Giao thức này biến mô hình thành một "bộ não cầm tay", sẵn sàng để lõi Camera 
#   nạp ngược lại vào RAM và chạy Real-time Inference mà không cần học lại.
# ==============================================================================
'''

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import pandas as pd
import joblib
import os

# ==========================================
# CẤU HÌNH TOÀN CỤC
# ==========================================
DATASET_PATH = 'data_output.csv'
TEST_SIZE_RATIO = 0.2  # 80% Train - 20% Test
RANDOM_SEED = 42

def phase1_load_and_split():
    """
    [PHASE 1] DATA INGESTION & SPLITTING
    """
    print("[INFO] PHASE 1: Đang nạp và phân mảnh dữ liệu...")
    
    # 1. Đọc dữ liệu bảng vào DataFrame (Tối ưu I/O bằng Pandas)
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        raise Exception(f"[FATAL ERROR] Không tìm thấy file {DATASET_PATH}. Hãy chắc chắn luồng ETL đã chạy thành công.")

    # Kiểm tra tính toàn vẹn của cấu trúc (1 cột Label + 40 cột Tọa độ = 41 cột)
    if df.shape[1] != 41:
        raise ValueError(f"[WARNING] Cấu trúc dữ liệu bất thường. Kì vọng 41 cột, nhưng nhận được {df.shape[1]} cột.")

    # 2. Bóc tách Không gian Đặc trưng (X) và Vector Đích (y)
    X = df.drop('Label', axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE_RATIO, 
        stratify=y, 
        random_state=RANDOM_SEED
    )

    # Xuất báo cáo cấu trúc không gian
    print(f"[SUCCESS] Phân mảnh hoàn tất:")
    print(f" - Tổng số mẫu: {len(df)}")
    print(f" - Tập huấn luyện (Train): {len(X_train)} mẫu ({100 - TEST_SIZE_RATIO*100}%)")
    print(f" - Tập kiểm định (Test): {len(X_test)} mẫu ({TEST_SIZE_RATIO*100}%)")
    
    return X_train, X_test, y_train, y_test

def phase2_and_3_train_with_gridsearch(X_train, y_train):
    """
    [PHASE 2 & 3] KHỞI TẠO MA TRẬN TÌM KIẾM VÀ ĐÓNG GÓI TRI THỨC
    """
    print("\n[INFO] PHASE 2 & 3: Kích hoạt Grid Search & Huấn luyện mô hình...")
    start_time = time.time()
    rf_base = RandomForestClassifier(random_state=RANDOM_SEED)

    param_grid = {
        'n_estimators': [50, 100, 150], #số lượng cây
        'max_depth': [10, 15, 20, None], #độ sâu tối đa của cây
        'min_samples_split': [2, 5, 10], #số mẫu tối thiểu để chia node
        'class_weight': ['balanced', 'balanced_subsample'] #cân bằng hoặc mặc định dữ liệu
    }

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        n_jobs=3,
        scoring='accuracy',
        verbose=2
    )

    print("[PROCESS] Đang duyệt qua không gian cấu hình. Quá trình này sẽ đẩy CPU lên 100%...")
    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"\n[SUCCESS] Huấn luyện hoàn tất trong {training_time:.2f} giây.")
    print(f"[RESULT] Cấu hình tối thượng (Best Parameters):")
    for key, value in grid_search.best_params_.items():
        print(f"   + {key}: {value}")
        
    print(f"[RESULT] Điểm kiểm định chéo (Cross-Validation Accuracy): {grid_search.best_score_ * 100:.2f}%")

    best_model = grid_search.best_estimator_
    
    return best_model

def phase4_evaluate_model(model, X_test, y_test):
    """
    [PHASE 4] VALIDATION & DIAGNOSTICS
    """
    print("\n[INFO] PHASE 4: Khởi động Suy luận mù (Blind Inference) trên tập Test...")
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[DIAGNOSTICS] ĐỘ CHÍNH XÁC TUYỆT ĐỐI (Accuracy): {acc * 100:.2f}%\n")
    
    print("[DIAGNOSTICS] BÁO CÁO PHÂN LỚP CHI TIẾT:")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    
    print("[PROCESS] Đang xuất bản đồ Ma trận Nhầm lẫn (Confusion Matrix)...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    
    plt.title('Khám nghiệm Ranh giới Quyết định (Confusion Matrix)', fontsize=14, pad=20)
    plt.xlabel('Nhãn Dự đoán bởi AI (Predicted Label)', fontsize=12)
    plt.ylabel('Nhãn Thực tế (True Label)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

def phase5_export_model(model, output_filename="ml_model.pkl"):
    """
    [PHASE 5] SERIALIZATION & EXPORT
    """
    print("\n[INFO] PHASE 5: Kích hoạt giao thức đóng băng và xuất xưởng...")
    
    try:
        # Thực thi nén và ghi nhị phân
        # Sử dụng nén (compress=3) để giảm thiểu dung lượng file trên đĩa
        # mà không làm ảnh hưởng đến tốc độ nạp (load) sau này.
        joblib.dump(model, output_filename, compress=3)
        
        # Giám sát vật lý tệp tin
        if os.path.exists(output_filename):
            file_size = os.path.getsize(output_filename) / (1024 * 1024)
            print(f"[SUCCESS] Đóng gói hoàn tất!")
            print(f" - Đường dẫn: {os.path.abspath(output_filename)}")
            print(f" - Kích thước 'bộ não': {file_size:.2f} MB")
            print("[SYSTEM] TẤT CẢ QUY TRÌNH ETL VÀ ML ĐÃ KẾT THÚC. HỆ THỐNG SẴN SÀNG CHO REAL-TIME INFERENCE.")
        else:
            print("[ERROR] Quá trình ghi đĩa thất bại, không tìm thấy file.")
            
    except Exception as e:
        print(f"[FATAL ERROR] Lỗi cấp hệ thống khi xuất file nhị phân: {e}")

if __name__ == "__main__":
    print("="*70)
    print(" KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN AI TRUNG TÂM (TRAINING PIPELINE) ")
    print("="*70)

    try:
        # [PHASE 1] Nạp và Phân mảnh không gian dữ liệu
        X_train, X_test, y_train, y_test = phase1_load_and_split()
        
        # [PHASE 2 & 3] Khởi tạo động cơ Grid Search và Ép xung học hỏi
        final_model = phase2_and_3_train_with_gridsearch(X_train, y_train)
        
        # [PHASE 4] Khám nghiệm pháp y thuật toán
        # LƯU Ý KỸ THUẬT: Khi biểu đồ Confusion Matrix hiện lên, luồng code sẽ bị TẠM DỪNG.
        # Bạn buộc phải soi chiếu xong và TẮT cửa sổ hình ảnh đó đi thì Phase 5 mới được kích hoạt.
        phase4_evaluate_model(final_model, X_test, y_test)
        
        # [PHASE 5] Đóng băng trạng thái và Xuất xưởng
        phase5_export_model(final_model, output_filename="ml_model.pkl")
        
        print("\n" + "="*70)
        print(" [HOÀN TẤT CHIẾN DỊCH] BỘ NÃO CHUỘT ẢO ĐÃ ĐƯỢC ĐÓNG GÓI THÀNH CÔNG ")
        print("="*70)

    except KeyboardInterrupt:
        print("\n[WARNING] Hệ thống nhận lệnh hủy khẩn cấp từ chỉ huy (Ctrl+C). Quá trình bị ngắt.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Cấu trúc sụp đổ trong quá trình thực thi: {e}")