# Blog: Hồi Quy và Chuỗi Thời Gian

## 1. Giới Thiệu

Nhóm bài toán này trả lời hai câu hỏi:
1. Hồi quy: dự đoán giá trị liên tục `Tool wear [min]`.
2. Chuỗi thời gian: dự báo xu hướng theo thứ tự vận hành.

Ý nghĩa thực tế:
- Không chỉ biết có hỏng hay không, mà còn biết mức độ mòn và xu hướng biến đổi.

## 2. Thiết Lập Thí Nghiệm

### 2.1 Hồi quy

Mô hình dùng trong `supervised.py`:
1. Linear Regression
2. RandomForestRegressor
3. GradientBoostingRegressor
4. XGBRegressor (nếu có)

Cấu hình chính:
- RF: `n_estimators=200`, `max_depth=10`
- GBR: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`
- XGB: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`

Chỉ số đánh giá:
1. MAE
2. RMSE
3. R2

### 2.2 Chuỗi thời gian

Do dataset không có timestamp thật, dự án dùng:
- `UDI` như time index gần đúng.

Thiết lập trong `forecasting.py`:
1. Temporal split theo thứ tự (`train_ratio=0.8`)
2. ARIMA với `order=(2,1,2)`
3. Lag-feature regression bằng GradientBoostingRegressor

Chỉ số đánh giá:
- MAE, RMSE, và thêm AIC/BIC cho ARIMA

## 3. Kết Quả Tổng Quan

### 3.1 Hồi quy

Nhận xét thường gặp:
1. Mô hình cây/boosting bắt quan hệ phi tuyến tốt hơn linear.
2. MAE cho biết sai số trung bình theo đơn vị phút rất trực quan.
3. RMSE phản ánh mức phạt mạnh với các lỗi lớn.

### 3.2 Chuỗi thời gian

1. ARIMA là baseline thống kê quan trọng để so sánh.
2. Lag-regression tận dụng tốt feature cảm biến nên thường linh hoạt hơn.
3. Temporal split giúp đánh giá gần thực tế hơn random split.

### 3.3 Giá trị ứng dụng

1. Dự đoán mòn dao giúp lập lịch thay dao chủ động.
2. Dự báo xu hướng giúp lên kế hoạch bảo trì theo ca/ngày.
3. Kết hợp với phân lớp tạo hệ cảnh báo đa tầng.

## 4. Kết Luận và Khuyến Nghị

### 4.1 Kết luận

- Hồi quy bổ sung chiều sâu cho quyết định bảo trì, không chỉ dừng ở nhãn hỏng.
- Chuỗi thời gian giúp theo dõi động thái hệ thống theo thời gian vận hành.

### 4.2 Khuyến nghị

1. Dùng MAE làm KPI vận hành dễ hiểu cho đội bảo trì.
2. Kiểm tra residual theo từng vùng giá trị để phát hiện bias mô hình.
3. Khi có timestamp thật trong tương lai, nâng cấp sang pipeline time series đầy đủ (seasonality, external covariates).
4. Kết hợp dự báo mòn dao với ngưỡng rủi ro để sinh khuyến nghị thay dao tự động.