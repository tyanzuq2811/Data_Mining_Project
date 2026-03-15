# Blog: Luật Kết Hợp (Association Rules - Apriori)

## 1. Giới Thiệu

Bài toán luật kết hợp trong đồ án nhằm trả lời câu hỏi:
- "Những tổ hợp điều kiện vận hành nào thường đi cùng trạng thái hỏng máy?"

Khác với phân lớp (dự đoán), Apriori tập trung vào khám phá tri thức:
- Không tối ưu để dự đoán xác suất trực tiếp.
- Rất mạnh để tạo insight dạng "IF ... THEN ...".

Trong ngữ cảnh bảo trì dự đoán, giá trị lớn nhất của luật kết hợp là:
1. Giải thích nguyên nhân theo tổ hợp điều kiện.
2. Hỗ trợ kỹ sư lập rule cảnh báo sớm trong vận hành.
3. Làm bằng chứng định tính để đối chiếu với kết quả phân lớp.

## 2. Thiết Lập Thí Nghiệm

### 2.1 Dữ liệu đầu vào

- Nguồn: AI4I 2020 Predictive Maintenance.
- Các biến số được rời rạc hóa thành mức thấp/trung bình/cao.
- Biến đích quan tâm: `Machine failure` và các failure modes.

### 2.2 Tiền xử lý cho Apriori

Vì Apriori cần dữ liệu nhị phân, pipeline thực hiện:
1. Chuyển từng biến số thành 3 cờ: low, normal, high.
2. One-hot cho `Type`.
3. Giữ các cột failure để truy vết luật theo lỗi.

Hàm dùng trong code:
- `FeatureBuilder.get_apriori_features()`
- `AssociationMiner.mine()`

### 2.3 Cấu hình thí nghiệm trong dự án

Theo `configs/params.yaml`:
- `min_support = 0.01`
- `min_confidence = 0.5`
- `min_lift = 1.5`
- `max_len = 4`

Ý nghĩa lựa chọn:
1. Support 1% để tránh luật quá hiếm và khó hành động.
2. Confidence 50% để bảo đảm độ tin cậy tối thiểu.
3. Lift > 1.5 để giữ luật có ý nghĩa hơn ngẫu nhiên.
4. Độ dài luật tối đa 4 để dễ diễn giải với kỹ sư vận hành.

### 2.4 Tiêu chí đánh giá

Không dùng Accuracy như bài toán phân lớp. Thay vào đó đọc:
1. Support: luật xuất hiện đủ thường xuyên không.
2. Confidence: mức tin cậy của luật.
3. Lift: độ mạnh liên kết thực sự.

## 3. Kết Quả Tổng Quan

### 3.1 Những kiểu luật nổi bật

Các luật nổi bật thường có dạng:
- Tổ hợp tải cao (`Torque` cao, `power` cao) + mòn dao cao.
- Điều kiện tản nhiệt xấu (`temp_diff` bất lợi) đi kèm failure.

Điều này phù hợp với hiểu biết kỹ thuật:
1. Máy quá tải kéo dài làm tăng stress cơ học.
2. Tản nhiệt kém làm tăng nguy cơ lỗi nhiệt.

### 3.2 Giá trị thực tế của kết quả

1. Luật kết hợp giúp chuyển kết quả dữ liệu thành ngôn ngữ vận hành.
2. Có thể dùng làm luật cảnh báo trong dashboard.
3. Dễ truyền đạt cho đội bảo trì hơn mô hình hộp đen.

### 3.3 Giới hạn cần lưu ý

1. Luật mạnh chưa chắc là quan hệ nhân quả.
2. Dữ liệu hiếm có thể tạo luật không ổn định.
3. Nếu ngưỡng support quá cao sẽ bỏ sót luật hiếm nhưng quan trọng.

## 4. Kết Luận và Khuyến Nghị

### 4.1 Kết luận

- Apriori trong đồ án đóng vai trò công cụ "khai phá tri thức" hơn là "công cụ dự đoán".
- Kết quả giúp xác định tổ hợp điều kiện rủi ro cao một cách trực quan.

### 4.2 Khuyến nghị triển khai

1. Duy trì `lift` làm tiêu chí chính khi chọn luật đưa vào cảnh báo.
2. Kết hợp luật với mô hình phân lớp để giảm cảnh báo sai.
3. Rà soát luật theo chu kỳ (hàng tháng) vì phân bố vận hành có thể thay đổi.
4. Tạo danh sách "Top luật hành động" gồm 5-10 luật rõ ràng, ưu tiên luật có support đủ lớn và confidence cao.