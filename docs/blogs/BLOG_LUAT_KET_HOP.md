# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 1: Luật Kết Hợp (Association Rules)

## 1. Mục tiêu của tab này
Tab Luật kết hợp dùng để tìm các tổ hợp điều kiện thường đi cùng sự cố, theo dạng:

`Điều kiện A, B, C -> Machine failure`

Mục tiêu không phải dự báo xác suất cho từng mẫu như phân lớp, mà là rút ra tri thức vận hành để đặt cảnh báo sớm theo luật.

## 2. Cách đọc chỉ số
- Support: tỷ lệ luật xuất hiện trong toàn bộ dữ liệu.
- Confidence: xác suất vế phải xảy ra khi vế trái đã xảy ra.
- Lift: mức độ mạnh của quan hệ so với ngẫu nhiên; Lift > 1 là có ý nghĩa, Lift càng cao thì luật càng đáng chú ý.

## 3. Kết quả định lượng chính
Dựa trên bảng luật hiện tại, các luật mạnh nhất đều có hậu quả là `Machine failure`:

1. `Air_temp high + Process_temp normal + Speed low -> Machine failure`
	- Support = 0.009
	- Confidence = 0.3516
	- Lift = 10.37

2. `Air_temp high + Process_temp normal + Torque high -> Machine failure`
	- Support = 0.0076
	- Confidence = 0.2980
	- Lift = 8.79

3. `Air_temp high + Speed low + Tool wear high -> Machine failure`
	- Support = 0.0048
	- Confidence = 0.2963
	- Lift = 8.74

Nhận xét nhanh:
- Support không lớn (đặc trưng bài toán lỗi hiếm), nhưng Lift rất cao, nên đây là các luật có giá trị cảnh báo.

## 4. Ý nghĩa vận hành
- Khi điều kiện nhiệt - tốc độ - tải đồng thời xuất hiện, nguy cơ lỗi tăng mạnh.
- Rule có Lift > 10 nên được xem là tín hiệu rủi ro cấp cao.
- Rule có Lift từ 7 đến 10 phù hợp cho cảnh báo sớm để kiểm tra trước khi lỗi bùng phát.

## 5. Khuyến nghị hành động cụ thể
1. Thiết lập cảnh báo 2 cấp theo Lift:
	- Cấp 1 (critical): Lift > 10
	- Cấp 2 (warning): Lift > 7
2. Với các luật có antecedent chứa `Speed low` và `Torque high`, yêu cầu kiểm tra cơ khí trong ca hiện tại.
3. Với các luật chứa `Tool wear high`, ưu tiên thay dụng cụ sớm hơn lịch chuẩn.
4. Rà soát lại luật hàng tháng, loại luật có support quá thấp và không còn tái diễn.

## 6. Câu nói chuẩn để dùng trong báo cáo
"Tab Luật kết hợp cho thấy các tổ hợp điều kiện nhiệt độ, tốc độ và tải có liên hệ mạnh với sự cố (Lift cao), từ đó cho phép thiết kế ngưỡng cảnh báo sớm theo luật thay vì chỉ dựa vào một biến đơn lẻ."
