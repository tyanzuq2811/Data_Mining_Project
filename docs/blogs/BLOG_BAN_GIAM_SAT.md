# Series Blog: Khám Phá Tri Thức Từ Dữ Liệu Predictive Maintenance - Phần 4: Bán Giám Sát (Semi-Supervised)

## 1. Giới Thiệu
### Ý tưởng cực dễ hiểu (Học bài gốc để tự điền đáp án mạo danh)
Trong một nhà máy AI4I, thông số quay, nhiệt độ nhả về từ hệ thống cảm biến (Sensor) đếm tới hàng vạn dòng. Tuy nhiên để biết một cái mốc máy lúc 12:00 "Hư hay Sống" thì phải có anh kỹ sư nhập tay vào File Excel (Gọi là Nhãn - Label). Nhưng chẳng ai rảnh để canh gác 10,000 dòng cả, vậy dữ liệu thu về bị **"Mồ Côi Đáp Án"**.
Học bán giám sát sinh ra để gỡ bom: Đó là 1 học sinh giỏi được dạy kiến thức lõi thông qua 10 quyển sách có điểm sẵn, tự đúc kết kinh nghiệm rồi dũng cảm lấy bút ghi "Đáp án giả / Nhãn mạo danh (Pseudo-label)" cho 90 quyển sách trống rỗng chưa ai thèm đánh dấu. Biến đống tài liệu mồ côi thành Mỏ vàng khổng lồ.

## 2. Giải Thích Thuật Toán & Tham Số (Kèm Phân Tính AI4I)
Thuật toán ta vận hành có tên gọi **Self-Training Classifier (Máy phân loại tự huấn luyện)**. Nó ôm bên trong một bộ não (Nhân) tự chọn: Ta chọn dùng trái tim của LightGBM hoặc Random Forest đặt vào.
- **Cách nó xoay vòng:** Ăn trước 10% Nhãn chuẩn. Cầm kiến thức lao vào 90% dòng trống còn lại, nếu nó nhận diện được 1 dòng: *"Tao cực kì tự tin cái mức chênh nhiệt 8.7K này là Máy sẽ Hỏng với xác suất khẳng định 98%"*. Thuật toán sẽ ngay lập tức phết chữ "Hỏng" vào ô trống, tự biến đó thành tài sản huấn luyện ở Vòng lặp thứ 2.
- **Tham số `Threshold` (Ngưỡng tự tin/Ngưỡng cảnh giác):** Đây là vòng kim cô quan trọng nhất. Nếu set ngưỡng thấp (0.6), học sinh sẽ láo toét, nhìn đâu cũng thấy ung thư (Máy hơi ấm ti đã vội quẹt nhãn "Hỏng" giả). Bắt buộc phải ép Ngưỡng Thật Cao mới cho phép gắn nhãn.

## 3. Thiết Lập Thí Nghiệm Trong Dự Án
- **Chiến lược "Che Mắt" (Masking):** Cố ý giả lập cảnh nghèo đói bằng cách Xoá đi từ 80% đến 90% cột `Machine failure` trong tập AI4I (Sửa nhãn thành -1 / Vô danh). Chỉ chừa lại đúng 10% sự giám sát.
- **Quy tắc đấu trường:** So găng trực tiếp giữa một bên là **Supervised Base (Mô hình lười)** (Kẻ thà chấp nhận chỉ học trên vỏn vẹn 10% dòng đã đánh nhãn và mặc kệ dòng trống) VÀ một bên là **Self-Training** (Ăn cố 10% dòng rồi nảy nhãn ảo cho 90% phần còn lại).

## 4. Kết Quả Tổng Quan
Trích xuất từ kết xuất thực tế của pipeline:
- **Đấu sĩ Ngưỡng cao (`Threshold = 0.95`):** Với lệnh răn đe "Xác suất > 95% mới được phết Mực dán mác mạo danh", mô hình Self-Training cẩn trọng dò dẫm trong ban đêm. Sau khi quẹt thành công nhiều đợt Pseudo-label, đồ thị chỉ số F1 lập tức phóng vụt đi lên, bỏ xa kẻ Supervised Base ngoan cố. Nghĩa là việc thu lượm thêm dữ liệu không đầu đuôi thực sự có ích!
- **Thoát Thác Lỗi Hủy Diệt (Error Propagation):** Cảnh giác cao độ: Trong Insights báo cáo, nếu ta lơi lỏng `Threshold` xuống, Random Forest bị mờ mắt. Nó tưởng lầm một mũi khoan chỉ "Hơi mòn" là Lỗi, rồi lan truyền rải đinh căn bệnh đó cho hàng nghìn mũi khoan tiếp tục tự học mạo danh sau. Cuối cùng biến cái nhà máy ngập chìm trong Báo Động Giả (False Positive / False Alarm).

## 5. Kết Luận và Khuyến Nghị
Công thức cứu rỗi khi Nhà máy quá lười thuê "Nô lệ đánh nhãn dữ liệu".
- **Khuyến Nghị Hiện Trường:** Luôn luôn khảm thuật toán Self-Training vào đuôi dữ liệu khuyết. Ép chặt tham số `Threshold >= 0.95`. Bên cạnh đó cài đặt một chuông báo cáo phụ: Những máy móc bị AI phết cờ hỏng với xác suất mập mờ (Chỉ nằm viền 70-80%), phải đẩy về cho Human-in-the-loop (Con người) tự tay kiểm duyệt đối chứng.
- Rời mắt khỏi những nhãn dán "Hỏng/Sống" khô khan, để chuẩn bị đặt câu hỏi mới: **Bao Chờ Thì Hỏng?** ở phần **Hồi Quy & Chuỗi thời gian** siêu bá đạo ngay kế sau.
