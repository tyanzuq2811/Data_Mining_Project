# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 5: Hồi Quy (Regression) & Chuỗi Thời Gian (Time-Series)

## 1. Mục tiêu của tab này
Tab Hồi quy/Chuỗi thời gian dùng để trả lời câu hỏi định lượng:
- Giá trị sẽ là bao nhiêu?
- Xu hướng sắp tới tăng hay giảm?

Đây là tab phục vụ dự báo định lượng và hỗ trợ lập kế hoạch bảo trì theo thời điểm.

## 2. Cách đọc chỉ số
- MAE: sai số tuyệt đối trung bình, càng thấp càng tốt.
- RMSE: giống MAE nhưng phạt mạnh lỗi lớn, càng thấp càng tốt.
- R2: mức độ giải thích biến thiên, càng gần 1 càng tốt.
- AIC/BIC: tiêu chí chọn mô hình chuỗi thời gian, càng thấp càng tốt trong cùng bài toán.

## 3. Kết quả định lượng chính
### 3.1. Hồi quy thường
- random_forest_reg: MAE=53.3288, RMSE=62.7010, R2=0.0791
- xgboost_reg: MAE=52.9623, RMSE=62.7075, R2=0.0790
- gradient_boosting_reg: MAE=53.0351, RMSE=62.7276, R2=0.0784

Nhận xét: R2 khoảng 0.079 là thấp, chưa đủ tin cậy để đưa vào quyết định vận hành.

### 3.2. Chuỗi thời gian
- ARIMA(2,1,2): MAE=54.5115, RMSE=62.9723, AIC=73044.33, BIC=73079.26
- GBR_lag_features: MAE=4.9306, RMSE=23.4048, R2=0.8625

Nhận xét: GBR_lag_features vượt trội rõ rệt cho dự báo ngắn hạn.

## 4. Ý nghĩa vận hành
- Hồi quy thường hiện mới mang tính tham khảo phân tích.
- Mô hình chuỗi thời gian với feature trễ đang đủ tốt để hỗ trợ cảnh báo xu hướng ngắn hạn.
- Có thể kết hợp với tab Phân lớp: phân lớp báo "có nguy cơ", chuỗi thời gian báo "mức độ và xu hướng".

## 5. Khuyến nghị hành động cụ thể
1. Dùng GBR_lag_features làm mô hình dự báo xu hướng ngắn hạn.
2. Chưa triển khai regression thường vào quyết định sản xuất cho đến khi nâng R2 lên mức mục tiêu.
3. Thiết lập theo dõi drift theo tuần (MAE, RMSE) để phát hiện suy giảm chất lượng.
4. Ưu tiên bổ sung đặc trưng theo thời gian và cửa sổ trễ để tăng độ chính xác.

## 6. Câu nói chuẩn để dùng trong báo cáo
"Tab Hồi quy/Chuỗi thời gian cung cấp mô hình dự báo định lượng và xu hướng; trong kết quả hiện tại, mô hình chuỗi thời gian với đặc trưng trễ cho chất lượng tốt hơn rõ rệt và phù hợp hơn cho hỗ trợ quyết định ngắn hạn."

## 7. Hạn chế phương pháp (Limitations) và điều kiện áp dụng
Trong bộ dữ liệu AI4I hiện tại không có timestamp thực, nên dự án đang dùng `UDI` như một **temporal proxy** để sắp thứ tự quan sát. Cách làm này phù hợp cho mục tiêu nghiên cứu baseline, nhưng cần nêu rõ các giới hạn học thuật sau:

1. `UDI` là mã định danh, không phản ánh chính xác khoảng cách thời gian giữa các quan sát.
2. Nếu dữ liệu trộn nhiều bối cảnh vận hành (ca, tải, máy), quan hệ "liền kề theo UDI" có thể không còn mang ý nghĩa động học thực.
3. Kết quả time-series vì vậy nên được diễn giải là "xấp xỉ xu hướng theo thứ tự quan sát", chưa nên xem là dự báo thời gian thực chuẩn công nghiệp.

Khuyến nghị triển khai thực tế:
1. Ưu tiên thu thập timestamp thực hoặc runtime-hour theo từng máy.
2. Tái huấn luyện và đối chiếu lại mô hình sau khi có trục thời gian thật.
3. Trong giai đoạn chưa có timestamp, chỉ dùng kết quả chuỗi thời gian như lớp hỗ trợ quyết định, không dùng làm tín hiệu dừng máy độc lập.
