# Blog: Phân Lớp (Classification)

## 1. Giới Thiệu

Đây là bài toán trung tâm của đồ án:
- Dự đoán `Machine failure` (0/1) từ dữ liệu cảm biến.

Thách thức lớn nhất:
- Dữ liệu mất cân bằng nặng (failure ~3.39%).

Vì vậy chiến lược đánh giá không thể chỉ nhìn Accuracy.
Trong đồ án, trọng tâm là:
1. F1-score
2. PR-AUC
3. Recall cho lớp lỗi

## 2. Thiết Lập Thí Nghiệm

### 2.1 Dữ liệu và chia tập

1. Dùng dữ liệu đã preprocess + feature engineering.
2. Loại cột leakage khi train.
3. Chia train/test theo stratified split để giữ tỷ lệ lớp.
4. Chuẩn hóa đầu vào trước train.

### 2.2 Các mô hình được huấn luyện

Trong `src/models/supervised.py`:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost (nếu môi trường có cài)
5. LightGBM (nếu môi trường có cài)

### 2.3 Cấu hình chính đang dùng

Logistic Regression:
- `C=1.0`, `max_iter=1000`, `class_weight='balanced'`

Random Forest:
- `n_estimators=200`, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'`

Gradient Boosting:
- `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`

XGBoost:
- `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `scale_pos_weight=28.5`

LightGBM:
- `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `is_unbalance=True`

Cross-validation:
- StratifiedKFold, `cv=5`

### 2.4 Chỉ số đánh giá

1. F1: cân bằng Precision/Recall.
2. PR-AUC: phù hợp dữ liệu lệch lớp.
3. ROC-AUC: mức phân biệt tổng quát.
4. Precision/Recall: đánh đổi false alarm và miss.
5. Train time: chi phí tính toán.

## 3. Kết Quả Tổng Quan

### 3.1 So sánh tổng quan

Thông thường trong bài toán này:
1. Tree-based boosting (Gradient Boosting/XGBoost/LightGBM) cho kết quả mạnh.
2. Logistic Regression là baseline dễ giải thích, hiệu quả ổn định.
3. Random Forest cân bằng tốt giữa hiệu năng và độ ổn định.

### 3.2 Góc nhìn vận hành

Trong bảo trì dự đoán:
- Bỏ sót lỗi (false negative) thường đắt hơn cảnh báo nhầm.

Vì vậy nên ưu tiên:
1. Recall và PR-AUC tốt.
2. F1 đủ cao để không cảnh báo tràn lan.

### 3.3 Ý nghĩa feature importance

Feature importance giúp trả lời:
- Mô hình dựa vào tín hiệu nào nhiều nhất.

Trong bối cảnh AI4I, các nhóm biến thường quan trọng:
1. Mài mòn dao cụ.
2. Mô-men xoắn và công suất.
3. Đặc trưng nhiệt và tương tác nhiệt-tải.

## 4. Kết Luận và Khuyến Nghị

### 4.1 Kết luận

- Phân lớp là thành phần ra quyết định chính trong hệ thống.
- Cách đánh giá đúng (F1/PR-AUC) quan trọng ngang hoặc hơn việc đổi thuật toán.

### 4.2 Khuyến nghị

1. Giữ metric ưu tiên là PR-AUC + F1 cho lớp lỗi.
2. Cân chỉnh threshold theo mục tiêu vận hành (an toàn vs false alarm).
3. Theo dõi drift dữ liệu và retrain định kỳ.
4. Kết hợp mô hình phân lớp với luật kết hợp để tăng khả năng giải thích.