# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 3: Phân Lớp (Classification)

## 1. Mục tiêu của tab này
Tab Phân lớp dự đoán trực tiếp trạng thái lỗi của máy (`Machine failure`: 0 hoặc 1). Đây là tab có thể chọn làm mô hình production chính vì cho đầu ra xác suất lỗi theo từng mẫu.

## 2. Vì sao chọn F1 làm chỉ số chính
Do dữ liệu lỗi hiếm (mất cân bằng lớp), chỉ dùng Accuracy sẽ dễ "đẹp giả". Vì vậy cần ưu tiên:
- Precision: giảm cảnh báo giả.
- Recall: giảm bỏ sót lỗi.
- F1: cân bằng Precision và Recall.

PR-AUC dùng làm chỉ số bổ sung rất quan trọng trong bối cảnh mất cân bằng lớp.

## 3. Kết quả định lượng chính
Kết quả so sánh các mô hình hiện tại:

1. Gradient Boosting
   - F1 = 0.8413
   - Precision = 0.9138
   - Recall = 0.7794
   - PR-AUC = 0.8461

2. LightGBM
   - F1 = 0.8027
   - Precision = 0.7468
   - Recall = 0.8676
   - PR-AUC = 0.8892

3. XGBoost
   - F1 = 0.7971
   - Precision = 0.7857
   - Recall = 0.8088
   - PR-AUC = 0.8775

Kiểm chứng ổn định (CV): Gradient Boosting có `cv_f1_mean = 0.856` và `cv_f1_std = 0.0182`, cho thấy độ ổn định tốt.

## 4. Ý nghĩa vận hành
- Nếu ưu tiên cân bằng tổng thể giữa cảnh báo giả và bỏ sót lỗi: chọn Gradient Boosting.
- Nếu ưu tiên bắt nhiều lỗi hơn (recall cao): LightGBM là phương án thay thế đáng cân nhắc.
- Quyết định cuối cùng nên gắn với chi phí vận hành: chi phí bỏ sót lỗi thường cao hơn chi phí kiểm tra dư.

## 5. Khuyến nghị hành động cụ thể
1. Chọn Gradient Boosting làm baseline production.
2. Theo dõi KPI theo thứ tự: F1 -> Recall -> PR-AUC.
3. Đặt ngưỡng cảnh báo tái huấn luyện nếu F1 < 0.80 hoặc Recall < 0.75.
4. Kiểm tra riêng theo nhóm Type/Cluster để giảm FN tại các nhóm rủi ro cao.

## 6. Câu nói chuẩn để dùng trong báo cáo
"Tab Phân lớp là mô hình dự báo chính để triển khai, với tiêu chí chọn mô hình dựa trên F1 nhằm cân bằng giữa bỏ sót lỗi (FN) và cảnh báo giả (FP), đồng thời kiểm chứng bằng PR-AUC trong bối cảnh dữ liệu mất cân bằng lớp."
