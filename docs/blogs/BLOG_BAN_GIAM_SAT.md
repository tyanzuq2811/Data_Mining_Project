# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 4: Bán Giám Sát (Semi-Supervised)

## 1. Mục tiêu của tab này
Tab Bán giám sát dùng khi dữ liệu có rất ít nhãn lỗi. Mục tiêu là đánh giá xem có thể tận dụng dữ liệu chưa gán nhãn để cải thiện mô hình hay không.

Các phương pháp đang so sánh:
- supervised_only
- self_training
- co_training
- label_spreading

## 2. Cách đọc chỉ số
- F1: chất lượng tổng thể trong bối cảnh mất cân bằng lớp.
- Precision/Recall: kiểm soát cảnh báo giả và bỏ sót lỗi.
- Pseudo-label risk: đánh giá rủi ro của nhãn giả (false alarm/miss).

## 3. Kết quả định lượng chính
Theo kết quả hiện tại:

1. Ở 5% nhãn:
	- supervised_only: F1 = 0.0845
	- self_training: F1 = 0.0000
	- label_spreading: F1 = 0.0563

2. Ở 10% nhãn:
	- supervised_only: F1 = 0.1266
	- self_training: F1 = 0.0000
	- co_training: F1 = 0.1500
	- label_spreading: F1 = 0.0652

3. Ở 20% nhãn:
	- supervised_only: F1 = 0.5902
	- self_training: F1 = 0.3967
	- co_training: F1 = 0.5312
	- label_spreading: F1 = 0.1414

4. Rủi ro pseudo-label của self-training:
	- 5-10% nhãn: n_pseudo_positive = 0, miss rate = 1.0
	- 20% nhãn: false alarm rate = 0.3721, miss rate = 0.8756

Nhận xét nhanh:
- Với cấu hình hiện tại, self-training chưa hiệu quả ở mức nhãn thấp.
- Co-training cải thiện rõ so với self-training ở mức 10-20% nhãn, nhưng vẫn thấp hơn supervised_only tại 20% nhãn.

## 4. Vì sao trong bài toán này Co-Training thường tốt hơn Self-Training?
Đây là điểm quan trọng để tránh hiểu nhầm khi đọc bảng kết quả:

1. Ở 5-10% nhãn, Self-Training gần như không gán được mẫu lỗi
- Kết quả cho thấy self_training có `n_pseudo_positive = 0` và F1 = 0.
- Nghĩa là mô hình quá bảo thủ, chủ yếu gán "normal", dẫn đến bỏ sót lỗi hàng loạt.

2. Co-Training dùng cơ chế đồng thuận 2 "view" đặc trưng
- Mẫu chỉ được gán nhãn giả khi cả 2 mô hình cùng dự đoán và cùng đủ tự tin.
- Cơ chế này giúp giảm lan truyền nhãn giả sai, nên ở 10-20% nhãn Co-Training cho F1 cao hơn Self-Training.

3. Vì sao có lúc Self-Training có thể nhỉnh hơn?
- Khi tỷ lệ nhãn tăng cao (ví dụ vùng 75% trong learning curve), dữ liệu có nhãn đã đủ mạnh.
- Lúc này lợi thế "lọc đồng thuận" của Co-Training giảm dần, trong khi Self-Training có thể tận dụng toàn bộ đặc trưng trong một mô hình duy nhất nên đôi khi nhỉnh nhẹ.

Kết luận ngắn gọn:
- Với cấu hình hiện tại và mức nhãn thấp-trung bình (<= 20%), Co-Training tốt hơn Self-Training.
- Tuy nhiên baseline ổn định nhất cho production hiện vẫn là supervised_only.

## 5. Ý nghĩa vận hành
- Bán giám sát trong bộ dữ liệu này chưa đủ ổn định để dùng làm mô hình chính.
- Khi nhãn quá ít, mô hình có xu hướng bảo thủ và bỏ sót nhiều lỗi.
- supervised_only vẫn là lựa chọn an toàn hơn trong giai đoạn hiện tại.
- co_training là phương án bán giám sát đáng cân nhắc hơn self_training khi cần thêm đường so sánh.

## 6. Khuyến nghị hành động cụ thể
1. Không dùng self-training cho production khi tỷ lệ nhãn < 15%.
2. Chỉ cân nhắc bán giám sát khi đạt >= 20% nhãn và có giám sát rủi ro pseudo-label.
3. Giữ supervised_only làm baseline cho kịch bản ít nhãn.
4. Nếu bắt buộc dùng bán giám sát, ưu tiên co_training trước self_training trong cấu hình hiện tại.
5. Áp dụng human-in-the-loop cho các mẫu có độ tự tin thấp để tích lũy nhãn thật.

## 7. Câu nói chuẩn để dùng trong báo cáo
"Tab Bán giám sát được dùng như kênh nghiên cứu khi thiếu nhãn; trong kết quả hiện tại, supervised-only vẫn ổn định hơn self-training ở mức nhãn thấp, do đó chưa khuyến nghị triển khai self-training làm mô hình production."
