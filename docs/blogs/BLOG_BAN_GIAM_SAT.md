# Blog: Bán Giám Sát (Semi-supervised Learning)

## 1. Giới Thiệu

Bài toán bán giám sát trả lời câu hỏi thực tế:
- Nếu chỉ có một phần nhỏ dữ liệu được gán nhãn, mô hình còn tốt không?

Trong nhà máy, gán nhãn lỗi thường:
1. Tốn thời gian.
2. Phụ thuộc chuyên gia.
3. Khó mở rộng nhanh.

Do đó, bán giám sát là hướng rất thực dụng cho triển khai thật.

## 2. Thiết Lập Thí Nghiệm

### 2.1 Kịch bản thí nghiệm

Đồ án mô phỏng 3 mức thiếu nhãn:
- 5%
- 10%
- 20%

Ở mỗi mức, so sánh 3 phương pháp:
1. Supervised-only baseline
2. Self-Training
3. Label Spreading

### 2.2 Cấu hình trong dự án

Theo `configs/params.yaml` và `src/models/semi_supervised.py`:

Self-Training:
- base estimator: RandomForestClassifier
- `threshold = 0.95`
- `max_iter = 30`

Label Spreading:
- `kernel = 'rbf'`
- `alpha = 0.2`
- `max_iter = 100`
- giới hạn train `n_max = 5000` để kiểm soát độ phức tạp

Baseline:
- RandomForest với class_weight balanced trên phần có nhãn

### 2.3 Chỉ số đánh giá

1. F1
2. Precision
3. Recall
4. Accuracy
5. ROC-AUC/PR-AUC (khi có xác suất)

Ngoài ra còn theo dõi:
- Số lượng pseudo-label sinh ra
- Tỷ lệ pseudo-label positive

## 3. Kết Quả Tổng Quan

### 3.1 Xu hướng điển hình

1. Khi tỷ lệ nhãn tăng từ 5% lên 20%, chất lượng tăng rõ.
2. Self-Training thường vượt baseline nếu threshold hợp lý.
3. Label Spreading phụ thuộc mạnh vào cấu trúc dữ liệu và scale.

### 3.2 Ý nghĩa vận hành

Kết quả cho thấy:
- Có thể khởi chạy hệ thống sớm dù nhãn còn ít.
- Sau đó tăng dần nhãn qua thời gian để cải thiện mô hình.

Đây là lộ trình phù hợp cho nhà máy mới số hóa dữ liệu.

### 3.3 Rủi ro

1. Pseudo-label sai có thể lan truyền lỗi học.
2. Threshold quá thấp làm nhiễu dữ liệu huấn luyện.
3. Label Spreading có thể nặng tính toán khi dữ liệu lớn.

## 4. Kết Luận và Khuyến Nghị

### 4.1 Kết luận

- Bán giám sát là chiến lược khả thi khi dữ liệu nhãn khan hiếm.
- Self-Training là lựa chọn thực dụng, dễ triển khai trong pipeline hiện tại.

### 4.2 Khuyến nghị

1. Bắt đầu với Self-Training và threshold cao (0.9-0.95).
2. Theo dõi chất lượng pseudo-label theo từng vòng.
3. Kết hợp active learning để ưu tiên gán nhãn mẫu khó.
4. Định kỳ so sánh lại với supervised-only để đảm bảo không trôi chất lượng.